package edu.utulsa.conversation.tm

import breeze.linalg._
import breeze.numerics.{abs, exp, log, pow}
import breeze.optimize._
import java.io.File

import edu.utulsa.conversation.text.{Corpus, Dictionary, Document}

import scala.util.control.Breaks._

class MixedMembershipTopicFlowModel
(
  override val numTopics: Int,
  override val words: Dictionary,
  override val documentInfo: Map[String, List[TPair]],
  override val wordInfo: Map[String, List[TPair]],
  val sigma: DenseMatrix[Double],
  val a: DenseMatrix[Double],
  val theta: DenseMatrix[Double]
) extends TopicModel(numTopics, words, documentInfo, wordInfo) {
  override protected def saveModel(dir: File): Unit = {
    csvwrite(new File(dir + "/a.mat"), a)
    csvwrite(new File(dir + "/sigma.mat"), sigma)
    csvwrite(new File(dir + "/theta.mat"), theta)
  }

  override def likelihood(corpus: Corpus): Double = ???
}

object MixedMembershipTopicFlowModel {
  def train(corpus: Corpus, numTopics: Int, numIterations: Int): MixedMembershipTopicFlowModel = ???
}

sealed class MMTFMOptimizer
(
  override val corpus: Corpus,
  override val numTopics: Int,
  val sigmaFactor: Double = 4.0
) extends TMOptimizer[MixedMembershipTopicFlowModel](corpus, numTopics) {

  import MathUtils._

  override def train(): MixedMembershipTopicFlowModel = ???

  val N = corpus.size
  val M = corpus.words.size

  // Parameters
  private val a: DenseMatrix[Double] = DenseMatrix.zeros[Double](K, K) // k x k
  private val sigma: DenseMatrix[Double] = normalize(DenseMatrix.rand[Double](K, M), Axis._1, 1.0) // k x k
  private val theta: DenseMatrix[Double] = DenseMatrix.zeros[Double](K, K) // k x w

  val nodes: Seq[DNode] = {
    val m = corpus.map { case (document: Document) =>
      document -> new DNode(document)
    }.toMap
    m.foreach { case (document, node) =>
      if(document.parent.isDefined)
        node.parent = m(document.parent.get)
      if(document.replies != null && document.replies.nonEmpty)
        node.children = document.replies.map(m)
    }
    m.values.toSeq
  }
  val roots: Seq[DNode] = nodes.filter(_.parent == null)
  val trees: Seq[DTree] = roots.map(new DTree(_))
  val replies: Seq[DNode] = nodes.filter(_.parent != null)

  val invSigma = DenseMatrix.zeros[Double](K, K)
  val sigmaA = DenseMatrix.zeros[Double](K, K)
  val I = DenseMatrix.eye[Double](K)
  val P = I :- (DenseMatrix.ones[Double](K, K) :/ K.toDouble)

  /**
    * Internal object to track topic distribution for individual documents.
    * @param document Corresponding document object.
    */
  class DNode(val document: Document) {
    def index: Int = document.index
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
        val lq = log(x) :+ log(theta(::, w))
        q(w) := normalize(exp(lq :- lse(lq)) :+ 1e-4, 1d)
      }
    }

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
        level.reverse.tail.foreach { case (nodes: Seq[DNode]) =>
          nodes.foreach((node) => {
            error += node.updateX(withConv, withConv)
            size += 1
          })
        }
        error = error / size
        iter = iter + 1
      }
    }

    def print(): Unit = {
      level.zipWithIndex.foreach { case (nodes, index) =>
        nodes.foreach((node) =>
          println({
            val (p, topic) = node.x.toArray.zipWithIndex.minBy(-_._1)
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
      }).sum
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
    a := (X.t \ Y.t).t
    disp(diag(a))
    a := a(::, *).map { case (vec) =>
      val add = -min(vec)
      normalize(vec :+ 1e-8 :+ (if(add > 0) add else 0d), 1d)
    }
    val error = sum(pow(oldA - a, 2.0))
//    println(s"     error: $error")
  }

  var inPLSA: Boolean = false
  def plsa(numIntervals: Int, updateTheta: Boolean = true) = {
    inPLSA = true
    for(i <- 1 to numIntervals) {
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
}
