package edu.utulsa.conversation.tm

import breeze.linalg._
import breeze.numerics.{abs, exp, log, pow}
import breeze.optimize._
import org.json4s._
import org.json4s.native.JsonMethods._
import org.json4s.native.Serialization
import org.json4s.native.Serialization.write
import java.io.File
import java.io.PrintWriter

import edu.utulsa.conversation.text.{Corpus, Document}

import scala.util.control.Breaks._

class MMCTMOptimize(val N: Int, val M: Int, val K: Int, val sigma: DenseMatrix[Double], val a: DenseMatrix[Double], val theta: DenseMatrix[Double])(val corpus: Corpus) extends MathUtils {
  var nodes: Seq[DNode] = null
  var roots: Seq[DNode] = null
  var trees: Seq[DTree] = null
  var replies: Seq[DNode] = null

  val invSigma = DenseMatrix.zeros[Double](K, K)
  val sigmaA = DenseMatrix.zeros[Double](K, K)
  val I = DenseMatrix.eye[Double](K)
  val P = I :- (DenseMatrix.ones[Double](K, K) :/ K.toDouble)
  class DNode(val index: Int, val document: Document) {
    var parent: DNode = null
    var children: Seq[DNode] = Seq()

    lazy val size: Int = this.children.foldLeft(1)((s,n) => s + n.size)
    // Optimal assignment stuff
    lazy val count: Map[Int, Int] = document.count.toMap
    val q: Map[Int, DenseVector[Double]] = {
      document.count.map { case (w, c) =>
        w -> DenseVector.zeros[Double](K)
      }.toMap
    }
    def updateQ() = {
      document.count.foreach { case (w, c) =>
        val lq = (log(x) :+ log(theta(::, w)))
        q(w) := normalize(exp(lq :- lse(lq)) :+ 1e-4, 1d)
      }
    }

    // val sumQ: Term[DenseVector[Double]] = Term {
    //   q().map(_._2).fold(DenseVector.zeros[Double](K))(_ + _)
    // }

    def f(y: DenseVector[Double] = this.x): Double = {
      val p1: Double =
        if(q.size <= 0)
          0d
        else
          sum(q.map { case (w, qi) => qi :* (log(y) :+ log(theta(::, w))) }.reduce(_ + _))
      val p2: Double = children
        .map { case (c: DNode) =>
          val d = c.x - a * y
          d.t * invSigma * d * -0.25
        }
        .foldLeft(0d)(_ + _)
      val p3: Double =
        if(parent != null) {
          val d = y - (a * (parent.x))
          d.t * invSigma * d * -0.25
        }
        else
          0d
      // if(index == 2) {
      //   println(f"$p1%4.3f $p2%4.3f $p3%4.3f")
      //   println(x)
      // }
      // p1 :+ p2 :+ p3
      p1 :+ p2 :+ p3
    }

    val x: DenseVector[Double] = {
      val tmp = DenseVector.rand(K)
      tmp :/ sum(tmp)
    }

    def f(x: DenseVector[Double], q: DenseVector[Double], b: DenseVector[Double], H: DenseMatrix[Double]): Double = {
      (-sum(q :* log(x))) :+ (x.t * H * x :* 0.5) :- b.t * x
    }

    // def alphaGrad(alpha: Double, q: DenseVector[Double], b: DenseVector[Double], A: DenseMatrix[Double], d: DenseVector[Double], x: DenseVector[Double]): Double = {
    //   sum((q :* d) :/ (x :+ (alpha :* d))) + (b dot d) - (d.t * A * x) - (d.t * A * d * alpha)
    // }

    // def alphaHess(alpha: Double, q: DenseVector[Double], d: DenseVector[Double], dAd: Double): Double = {
    //   sum(q :* pow(d :/ (x :+ (alpha :* d)), 2d)) :- dAd
    // }

    def grad(y: DenseVector[Double], q: DenseVector[Double], b: DenseVector[Double], H: DenseMatrix[Double]): DenseVector[Double] = {
      - (q :/ y) :+ (H * y) :- b
    }

    def gphi(y: DenseVector[Double], alpha: Double, d: DenseVector[Double], q: DenseVector[Double], b: DenseVector[Double], H: DenseMatrix[Double]): Double = {
      (-P * grad(y :+ (d * alpha), q, b, H)) dot y
    }

    def updateX(useParent: Boolean = true, useChildren: Boolean = true): Double = {
      // println("####")
      val origX = x.copy
      val newX = x.copy
      var error = 100.0
      var iter = 0
      val qs = q.map { case (w, q) =>
        q :* count(w).toDouble
      }
        .fold(DenseVector.zeros[Double](K))(_ + _)

      if(inPLSA) {
        this.x := normalize(qs, 1d)
        return norm(origX - this.x)
      }
      val ZERO = DenseVector.zeros[Double](K)
      val ZEROM = DenseMatrix.zeros[Double](K, K)
      val b1 =
        if(useChildren)
          sigmaA.t * children.map(_.x).fold(ZERO)(_ + _)
        else
          ZERO
      val b2 =
        if(parent != null && useParent)
          sigmaA * parent.x
        else
          ZERO
      val b = b1 :+ b2
      val c1 =
        if(useChildren)
          (sigmaA.t * a) :* children.length.toDouble
        else
          ZEROM
      val c2 =
        if(parent != null && useParent)
          invSigma
        else
          ZEROM
      val c = c1 :+ c2

      val gamma = 0.0
      val oldD = DenseVector.zeros[Double](K)

      // println("#" * 70)
      // println("ORIGINAL")
      // disp(this.x)
      breakable {
        while(error > 1e-3 && iter < 100) {
          val oldX = newX.copy

          val g = grad(newX, qs, b, c)
          val newD = (- P * g)
          val d = gamma * oldD + (1-gamma) * newD
          val xPhi = f(newX, qs, b, c)

          val dnorm = norm(d)
          if(dnorm < 1e-4)
            break
          d := d / dnorm

          val maxAlphas = (-oldX :/ d).toArray.filter((i) => !i.isNaN && i >= 0)
          var maxAlpha = 0d
          if(maxAlphas.isEmpty)
            break
          else
            maxAlpha = min(maxAlphas)
          d := d * maxAlpha :/ 100.0

          val lineSearch = new BacktrackingLineSearch(xPhi,
            growStep = 1.2,
            maxAlpha = 100.0
          )
          val func = new DiffFunction[Double] {
            def calculate(alpha: Double) = {
              (f(newX :+ (d * alpha), qs, b, c), gphi(newX, alpha, d, qs, b, c))
            }
          }
          // disp(newX)
          val alpha: Double = {
            try {
              lineSearch.minimize(func, 0.5)
            }
            catch {
              case _: LineSearchFailed => {
                println("    search failed")
                1e-3
              }
              case _: StepSizeUnderflow => {
                // println("    underflow")
                0d
              }
              case _: StepSizeOverflow => {
                println("    overflow")
                disp(oldX)
                disp(qs)
                disp(b)
                disp(c)
                max(maxAlpha - 1e-4, 0d)
              }
            }
          }
          newX := normalize(oldX :+ (d * alpha), 1d)
          // val (oldF, oldG): (Double, Double) = (f(oldX, qs, b, c), norm(P * grad(oldX, qs, b, c)))
          // val (newF, newG): (Double, Double) = (f(newX, qs, b, c), norm(P * grad(newX, qs, b, c)))
          // println(s" iter: $iter")
          // println(s" f(oldX) = $oldF")
          // println(s" f(newX) = $newF")
          // println(f"$alpha%8.4f  $newF%8.4f  $newG%8.4f")

          error = alpha
          iter = iter + 1
          oldD := newD
        }
      }

      this.x := newX

      // if(index == 2)
      //   disp(this.x)

      if(f(origX, qs, b, c) < f(newX, qs, b, c) - 1e-4)
        println(s"what! $iter ${f(origX, qs, b, c)} ${f(newX, qs, b, c)}")
      return norm(newX - origX)
    }
  }

  class DTree(val root: DNode) {
    private def expand(nodes: Seq[DNode]): Seq[Seq[DNode]] = {
      if(nodes.length <= 0)
        Seq()
      else {
        val children = nodes.flatMap(_.children)
        Seq(nodes) ++ expand(children)
      }
    }
    lazy val level: Seq[Seq[DNode]] = expand(Seq(root))
    lazy val size: Int = root.size
    def updateX(withConv: Boolean): Unit = {
      var error = 100.0
      var size = 0.0
      var iter = 0
      if(inPLSA) {
        level.foreach { case (nodes: Seq[DNode]) =>
          nodes.foreach((node) =>
            error = error + node.updateX(withConv, withConv))
        }
        return
      }
      while(error >= 1e-2 && iter < 3) {
        error = 0
        size = 0
        level.foreach { case (nodes: Seq[DNode]) =>
          nodes.foreach((node) => {
            error += node.updateX(withConv, withConv)
            size += 1
          })
        }
        // if(verbose) this.print()
        level.reverse.tail.foreach { case (nodes: Seq[DNode]) =>
          nodes.foreach((node) => {
            error += node.updateX(withConv, withConv)
            size += 1
          })
        }
        error = error / size
        // println(s"     error (${root.index}): $error")
        // if(verbose) this.print()
        iter = iter + 1
      }
    }

    def print(): Unit = {
      level.zipWithIndex.foreach { case (nodes, index) =>
        nodes.foreach((node) =>
          println({
            val (p, topic) = node.x.toArray.zipWithIndex.sortBy(-_._1).head
            val pc = 100*p
            val v = "[ " + node.x.toArray.map((d) => f"$d%2.3f ").mkString + "]"
            f"$v"
          })
        )
      }
    }
  }

  def qStep() = {
    println("   q")
    nodes.par.foreach { case (node) =>
      node.updateQ()
    }
  }

  def xStep(withConv: Boolean = true) = {
    println("   x")
    trees.zipWithIndex.foreach { case (tree, index) =>
      tree.updateX(withConv)
    }
    // val f = nodes.map(_.f()).reduce(_ + _)
    // println(f"  f(X) = $f%5.8f")
  }

  // Can update this once
  def thetaStep() = {
    println("   theta")
    theta := DenseMatrix.zeros[Double](K, M)
    nodes.foreach { case (node) =>
      node.q.map { case (w, q) =>
        theta(::, w) :+= q
      }
    }
    theta := normalize(normalize(theta, Axis._1, 1.0) :+ 1e-8, Axis._1, 1.0)
  }

  var sigmaFactor: Double = 1.0
  // Need to loop these steps
  def sigmaStep() = {
    println("   sigma")
    // First method: Solve for full covariance matrix
    // val s = replies
    //   .map((node) => {
    //     val d = node.x - a * node.parent.x
    //     d * d.t
    //   })
    //   .reduce(_ + _)
    // sigma := s / replies.length.toDouble
    // if(rank(sigma) < K) {
    //   println("Sigma isn't positive definite. :(")
    //   sigma := sigma :+ DenseMatrix.eye[Double](K)
    // }
    // println(sigma)
    // Second method: Diagonal matrix, all same value
    val s: Double = replies
      .map((node) => {
        val d = node.x - a * node.parent.x
        d dot d
      })
      .reduce(_ + _)
    sigma := diag(DenseVector.fill(K) { s / (sigmaFactor * replies.length.toDouble) })
    // Third method: static value
    // sigma := DenseMatrix.eye[Double](K) :/ 3.5
    println(s"  sigma: ${sigma(0,0)}")
    println(s"  sigma-factor: $sigmaFactor")
  }

  def aStep() = {
    println("   a")
    val oldA = a.copy
    val X = replies.map((node) => node.parent.x * node.parent.x.t).reduce(_ + _)
    val Y = replies.map((node) => node.x * node.parent.x.t).reduce(_ + _)
    a := ((X.t) \ (Y.t)).t
    // println("A (unnormalized)")
    // println(a)
    disp(diag(a))
    // a := normalize(normalize(abs(a), Axis._0, 1.0) :+ 1e-8, Axis._0, 1.0)
    a := a(::, *).map { case (vec) =>
      val add = -min(vec)
      normalize(vec :+ 1e-8 :+ (if(add > 0) add else 0d), 1d)
    }
    val error = sum(pow(oldA - a, 2.0))
    println(s"     error: $error")
    // println("A (corrected)")
    // a(*, ::).foreach{ case (row) =>
    //   println("[ " + row.toArray.map((d) => f"$d%2.3f ").mkString + "]")
    // }
    // println(rank(a))
  }

  var inPLSA: Boolean = false
  def plsa(numIntervals: Int, updateTheta: Boolean = true) = {
    inPLSA = true
    for(i <- (1 to numIntervals)) {
      println(s"  Step $i")
      qStep()
      xStep(false)
      if(updateTheta)
        thetaStep()
    }
    inPLSA = false
  }

  def step(interval: Int) = {
    qStep()

    thetaStep()

    // for(i <- (1 to 3)) {
    aStep()

    val s: Double = replies
      .map((node) => {
        val d = node.x - a * node.parent.x
        d dot d
      })
      .reduce(_ + _)
    println(s"  avg. diff: ${s/replies.length}")

    // sigmaStep()
    updateParams()
    xStep()
    // }
  }

  private def updateParams() {
    invSigma := inv(sigma)
    sigmaA := sigma \ a
  }

  def initialize() = {
    println("Creating temp param matrices")
    val nmap = corpus.documents.zipWithIndex.map { case (document: Document, index: Int) =>
      document -> new DNode(index, document)
    }.toMap
    nmap.foreach { case (document, node) =>
      if(document.parent != null)
        node.parent = nmap(document.parent)
      if(document.children != null && document.children.length > 0)
        node.children = document.children.map(nmap(_))
    }
    nodes = nmap.map(_._2).toSeq
    roots = nodes.filter(_.parent == null)
    trees = roots.map(new DTree(_))
    replies = nodes.filter(_.parent != null)
    println(s"   ${roots.length} conversations")
    println(s"   ${replies.length} replies")
  }
}


/****************************************************************
 * MIXED MEMBERSHIP CONVERSATIONAL TOPIC MODEL
 ****************************************************************/

class MMCTopicModel(override val corpus: Corpus, saveDir: String) extends TopicModel(corpus) with MathUtils {
  private val dsave = if(!saveDir.endsWith("/")) saveDir + "/" else saveDir
  private var N: Int = corpus.numDocuments
  private var M: Int = corpus.numWords

  // Parameters
  // private var xmat: DenseMatrix[Double] = null   // k x n
  private var sigma: DenseMatrix[Double] = null  // k x k
  private var a: DenseMatrix[Double] = null      // k x k
  private var theta: DenseMatrix[Double] = null  // k x w

  private var sigmaFactor: Double = 4.0
  def setSigmaFactor(sigmaFactorValue: Double): this.type = {
    this.sigmaFactor = sigmaFactorValue
    this
  }

  private def randPDM(k: Int): DenseMatrix[Double] = {
    val a = lowerTriangular(DenseMatrix.rand[Double](k, k))
    a * a.t + diag(DenseVector.rand[Double](k) :* 10.0)
  }

  var optim: MMCTMOptimize = null
  private def maybeLoadPLSA(): Boolean = {
    val thetaFile = new File(dsave + "plsa/theta.mat")
    if(thetaFile.exists()) {
      println("Reloading PLSA...")
      val newTheta = csvread(thetaFile)
      if(theta.rows == newTheta.rows) {
        optim.plsa(15, updateTheta=false)
        true
      }
      else
        false
    }
    else {
      false
    }
  }
  private def initialize() = {
    println(s" $N documents")
    println(s" $M words")
    println(s" $K topics")

    println(" Building param matrices...")
    println(" A")
    a = DenseMatrix.zeros[Double](K, K)
    println(" theta")
    theta = normalize(DenseMatrix.rand[Double](K, M), Axis._1, 1.0)
    // println(" xmat")
    // xmat = normalize(DenseMatrix.rand[Double](K, N), Axis._0, 1.0)
    println(" sigma")
    // sigma = randPDM(K)
    sigma = DenseMatrix.zeros[Double](K, K)

    println(" optimization")
    optim = new MMCTMOptimize(N, M, K, sigma, a, theta)(corpus: Corpus)
    optim.initialize()
    optim.sigmaFactor = this.sigmaFactor
    println("Done.")
  }

  def train(): this.type = {
    println("Training")
    initialize()
    // if(!maybeLoadPLSA() || numIterations <= 0) {
      println("PLSA pre-step")
      optim.plsa(30)
      savePLSA(dsave + "plsa/")
      saveInfo(dsave + "plsa/")
    // }
    println("Main optimization step")
    optim.aStep()
    optim.sigmaStep()
    for(i <- (1 to numIterations)) {
      println(s"Step $i")
      optim.step(i)
    }
    this
  }

  sealed case class DTopicResult(documents: Map[String, Array[Double]], numTopics: Int)
  sealed case class DWordResult(documents: Map[String, List[WResult]], numTopics: Int)
  sealed case class WResult(w: String, t: Int, p: Double)
  private def savePLSA(dir: String): Unit = {
    new File(dir).mkdirs()
    csvwrite(new File(dir + "theta.mat"), theta)
  }
  private def saveInfo(dir: String): Unit = {
    implicit val formats = Serialization.formats(NoTypeHints)
    val topics = DTopicResult(
      optim.nodes.map { case (node) =>
          node.document.id -> node.x.toArray
      }.toMap,
      K
    )
    val words= DWordResult(
      optim.nodes.map { case (node) =>
        node.document.id -> node.q.map {
          case (i, w) =>
            w.toArray.zipWithIndex.sortBy(-_._1).take(1)
              .map((t) => WResult(corpus.words(i), t._2, t._1)).head
        }.toList
      }.toMap,
      K
    )

    Some(new PrintWriter(dir + "documents.json"))
      .foreach { (p) => p.write(write(topics)); p.close() }
    Some(new PrintWriter(dir + "words.json"))
      .foreach { (p) => p.write(write(words)); p.close() }
    corpus.save(dir)
  }

  def save(dir: String): Unit = {
    val d = dsave + "mmctm/"
    new File(d).mkdirs()
    savePLSA(d)
    csvwrite(new File(d + "a.mat"), a)
    csvwrite(new File(d + "theta.mat"), theta)
    csvwrite(new File(d + "sigma.mat"), sigma)
    // csvwrite(new File(d + "x.mat"), xmat)
    saveInfo(d)
  }
}

object MMCTopicModel {
  def apply(corpus: Corpus, saveDir: String) = new MMCTopicModel(corpus, saveDir)
}
