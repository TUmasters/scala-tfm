package edu.utulsa.conversation.tm

import breeze.linalg._
import breeze.numerics.{exp, log, log1p, pow}
import java.io.File

import org.json4s._
import org.json4s.native.JsonMethods._
import org.json4s.native.Serialization
import org.json4s.native.Serialization.write
import java.io.PrintWriter

import edu.utulsa.conversation.text.{Corpus, Dictionary, Document}
import edu.utulsa.conversation.util.{math, Param, ParamCounter}
import spire.math.poly.Term

trait UATFMParams {
  implicit val modelParams: this.type = this
  implicit val counters: (ParamCounter, ParamCounter) = (new ParamCounter(), new ParamCounter())
  val M: Int
  val K: Int
  val G: Int
  val pi: Array[DenseVector[Double]]
  val logPi: Param[Array[DenseVector[Double]]] = Param {
    pi.map(log(_))
  }(counters._2)
  val a: Array[DenseMatrix[Double]]
  val theta: DenseMatrix[Double]
  val logTheta: Param[DenseMatrix[Double]] = Param {
    log(theta)
  }(counters._2)
  val r: DenseMatrix[Double]
}

class UserAwareTopicFlowModel
(
  override val numTopics: Int,
  val numUserGroups: Int,
  override val corpus: Corpus,
  val authors: Dictionary,
  override val documentInfo: Map[String, List[TPair]],
  override val wordInfo: Map[String, List[TPair]],
  val pi: Array[DenseVector[Double]],
  val a: Array[DenseMatrix[Double]],
  val theta: DenseMatrix[Double]
) extends TopicModel(numTopics, corpus, documentInfo, wordInfo) with UATFMParams {
  val M: Int = corpus.words.size
  val K: Int = numTopics
  val G: Int = numUserGroups
  override val r: DenseMatrix[Double] = null

   override lazy val params: Map[String, AnyVal] = super.params ++ Map(
    "num-user-groups" -> numUserGroups
  )
  override protected def saveModel(dir: File): Unit = {
    csvwrite(new File(dir + "/theta.csv"), theta)
    for(i <- 0 until numUserGroups) {
      math.csvwritevec(new File(dir + f"/pi.g$i%02d.csv"), pi(i))
      csvwrite(new File(dir + f"/a.g$i%02d.csv"), a(i))
    }
  }
  override def likelihood(corpus: Corpus): Double = {
    val (trees: Seq[UATFMInfer.DTree], dnodes: Seq[UATFMInfer.DNode], unodes: Seq[UATFMInfer.UNode]) = UATFMInfer.build(corpus)
    UATFMInfer.eStep(trees, dnodes, unodes, G)
    trees.map(_.loglikelihood).sum
  }
}

class UATFMAlgorithm extends TMAlgorithm {
  val numUserGroups: Parameter[Int] = new Parameter(5)
  def setNumUserGroups(value: Int): this.type = numUserGroups := value
  val numIterations: Parameter[Int] = new Parameter(20)
  def setNumIterations(value: Int): this.type = numIterations := value
  val maxEIterations: Parameter[Int] = new Parameter(10)
  def setMaxEIterations(value: Int): this.type = maxEIterations := value

  def train(corpus: Corpus): UserAwareTopicFlowModel = {
    new UATFMOptimizer(corpus, $(numTopics), $(numUserGroups), $(numIterations), $(maxEIterations))
      .train()
  }
}

object UserAwareTopicFlowModel {
  def train
  (
    corpus: Corpus,
    numTopics: Int,
    numUserGroups: Int,
    numIterations: Int,
    maxEIterations: Int
  ): UserAwareTopicFlowModel = {
    new UATFMOptimizer(corpus, numTopics, numUserGroups, numIterations, maxEIterations)
      .train()
  }
}

class UATFMOptimizer
(
  override val corpus: Corpus,
  override val numTopics: Int,
  val numUserGroups: Int,
  val numIterations: Int,
  val maxEIterations: Int
) extends TMOptimizer[UserAwareTopicFlowModel](corpus, numTopics) with UATFMParams {

  import math._

  override def train(): UserAwareTopicFlowModel = {
//    println("Initializing...")
    (1 to numIterations).foreach { (interval) =>
      print(s" (t $interval)")
//      println("  - E step")
      eStep(interval)
//      println("  - M step")
      mStep(interval)
    }
    println()
    val d: Map[String, List[TPair]] = dnodes.map((node) =>
      node.document.id ->
        (!node.z).data.zipWithIndex.map { case (p, i) => TPair(p, i) }.sortBy(-_.p).toList
    ).toMap
    val w: Map[String, List[TPair]] = (0 until M).map((w) =>
      corpus.words(w) ->
        theta(w, ::).t.toArray.zipWithIndex.map { case (p, i) => TPair(p, i) }.sortBy(-_.p).toList
    ).toMap
    new UserAwareTopicFlowModel(numTopics, numUserGroups, corpus, corpus.authors, d, w, pi, a, theta)
  }

  val N: Int = corpus.size
  val M: Int = corpus.words.size
  val U: Int = corpus.authors.size
  val G: Int = numUserGroups

  val (trees, dnodes, unodes) = UATFMInfer.build(corpus)

  class UATFMParams
  (
    val pi: Array[DenseVector[Double]],
    val a: Array[DenseMatrix[Double]],
    val theta: DenseMatrix[Double]
  ) {
    import math._

    val id: Int = rand.nextInt()

    /** Track these values just in case we need them **/
    lazy val logPi: Array[DenseVector[Double]] = pi.map(pig => log(pig))
    lazy val logA: Array[DenseMatrix[Double]] = a.map(ag => log(ag))
    lazy val logTheta: DenseMatrix[Double] = log(theta)
  }

  val roots: Seq[Int] = trees.map(_.root.index)

  /** MODEL PARAMETERS **/
  val pi: Array[DenseVector[Double]]    = (1 to G).map((i) => normalize(DenseVector.rand(K))).toArray // k x g
  val a: Array[DenseMatrix[Double]]     = (1 to G).map((i) => normalize(DenseMatrix.rand(K, K), Axis._1, 1.0)).toArray // g x k x k
  val theta: DenseMatrix[Double]        = normalize(DenseMatrix.rand(M, K), Axis._0, 1.0) // m x k

  /** LATENT VARIABLE ESTIMATES **/
  val q: DenseMatrix[Double]     = normalize(DenseMatrix.rand[Double](K, N), Axis._1, 1.0) // k x n
  val r: DenseMatrix[Double]     = normalize(DenseMatrix.rand[Double](G, U), Axis._1, 1.0) // g x u
  val qr: Array[DenseMatrix[Double]] = (1 to G).map((i) => DenseMatrix.zeros[Double](K, N)).toArray // g x k x n

  /** USEFUL INTERMEDIATES **/
  val responses: DenseVector[Double] = {
    val m = DenseVector.zeros[Double](N)
    dnodes.foreach{ case (node) =>
      m(node.index) = node.children.length.toDouble
    }

    m
  } // n x 1
  // Reply matrix
  val b: CSCMatrix[Double] = {
    val builder = new CSCMatrix.Builder[Double](N, N)

    dnodes.foreach((node) =>
      node.children.foreach((child) =>
        builder.add(node.index, child.index, 1d)
      )
    )

    builder.result()
  }   // n x n
  /** Word occurrence matrix **/
  val c: CSCMatrix[Double] = {
    val builder = new CSCMatrix.Builder[Double](N, M)

    corpus.zipWithIndex.foreach { case (document, index) =>
      document.count.foreach { case (word, count) =>
        builder.add(index, word, count.toDouble)
      }
    }

    builder.result()
  }   // n x m

  protected def eStep(interval: Int): Unit = {
    UATFMInfer.eStep(trees, dnodes, unodes, G, q, r, qr)
  }

  protected def mStep(interval: Int): Unit = {
    Seq(
      () => {
        // Pi Maximization
        pi.zipWithIndex.foreach { case (pi_g, g) =>
          pi_g := normalize(roots.map((index) => qr(g)(::, index)).reduce(_ + _) :+ 1e-6, 1.0)
        }
      },
      () => {
        // A maximization
        a.zipWithIndex.foreach { case (a_g, g) =>
          a_g := normalize((q * b * qr(g).t) :+ 1e-6, Axis._1, 1.0)
        }
      },
      () => {
        // Theta maximization
        theta := normalize((q * c) :+ 1e-8, Axis._1, 1.0).t
      }
    ).par.foreach { case (step) => step() }

    // println(s"   ~likelihood: ${inference.logLikelihood()}")
  }
}


object UATFMInfer {
  def build(corpus: Corpus)(implicit params: UATFMParams, counters: (ParamCounter, ParamCounter)): (Seq[DTree], Seq[DNode], Seq[UNode]) = {
    val dnodes: Map[Document, DNode] = corpus.zipWithIndex.map { case (document, index) =>
      document -> new DNode(document, index)(params, counters._2)
    }.toMap

//    println(" - Creating author nodes...")
    val unodes: Map[Int, UNode] = corpus.documents.map(_.author).toSet.map((author: Int) =>
      author -> new UNode(author)(params, counters._1)
    ).toMap

//    println(" - Mapping replies...")
    dnodes.foreach { case (document, node) =>
      node.parent = document.parent match {
        case Some(key) => dnodes(key)
        case None => null
      }
      node.children = document.replies.map(dnodes(_)).toSeq
    }

//    println(" - Mapping users to authors...")
    corpus.groupBy(_.author).foreach { case (author: Int, documents: List[Document]) =>
      unodes(author).documents = documents.map(dnodes(_)).toSeq
      documents.foreach((document) => {
        dnodes(document).unode = unodes(author)
      })
    }

//    println(" - Creating trees...")
    val trees = dnodes.values
      .filter((node) => node.isRoot)
      .map((root) => new DTree(root))
      .toSeq

    (trees, dnodes.values.toSeq, unodes.values.toSeq)
  }

  ////////////////////////////////////////////////////////////////
  // USER NODE
  ////////////////////////////////////////////////////////////////
  sealed class UNode(val user: Int)(implicit val params: UATFMParams, implicit val counter: ParamCounter) {

    import params._
    import math._

    var documents: Seq[DNode] = Seq()
    val r: Param[DenseVector[Double]] = Param {
      val oldR = r.value.copy
      val n = documents.map { case (node) =>
        if (node.isRoot)
          DenseVector[Double]((0 until G).map((g) =>
            log(pi(g) dot !node.z)
          ).toArray)
        else
          DenseVector[Double]((0 until G).map((g) =>
            log((!node.parent.z).t * a(g) * !node.z)
          ).toArray)
      }.reduce(_ + _)
      // Normalize
      val newR = exp(n :- lse(n))
      dist = norm(oldR - newR)
      newR
    }.default { DenseVector.rand[Double](G) }

    var dist: Double = G

    //  def update(): Unit = {
    //    val old_r = !r
    //    r(::, user) := !r
    //    if(old_r != null)
    //      dist = norm(old_r - r_g())
    //  }
  }


  ////////////////////////////////////////////////////////////////
  // DOCUMENT NODE
  ////////////////////////////////////////////////////////////////
  sealed class DNode(val document: Document, val index: Int)(implicit val params: UATFMParams, implicit val counter: ParamCounter) {

    import params._
    import math._

    var parent: DNode = _
    //var author: Int = -1
    var unode: UNode = _
    var children: Seq[DNode] = Seq()

    lazy val siblings: Seq[DNode] = {
      if (parent != null) {
        parent.children.filter((child) => child != this)
      }
      else {
        Seq()
      }
    }

    def isRoot: Boolean = {
      this.parent == null
    }

    /**
      * Computes log probabilities for observing a set of words for each latent
      * class.
      */
    val probW: Param[DenseVector[Double]] = Param {
      val result =
        if (document.words.nonEmpty)
          document.count
            .map { case (word, count) => (!logTheta)(word, ::).t :* count.toDouble }
            .reduce(_ + _)
        else
          DenseVector.zeros[Double](K)
      result
    }

    val lambdaMsg: Param[DenseVector[Double]] = Param {
      lse((0 until G).map((g) => lse(a(g), !lambda) :+ log((!unode.r)(g))).toArray)
    }

    val lambda: Param[DenseVector[Double]] = Param {
      // Lambda messages regarding likelihood of observing the document
      val msg1 = !probW
      // Lambda messages from children
      val msg2 =
        if (children.nonEmpty)
          children
            .map((child) => !child.lambdaMsg)
            .reduce(_ + _)
        else
          DenseVector.zeros[Double](K)
      msg1 + msg2
    }

    val tau: Param[DenseVector[Double]] = Param {
      implicit val tmp: UATFMParams = params
      if (parent == null) {
        lse((!logPi).zipWithIndex.map { case (pi_g, g) =>
          pi_g :+ log((!unode.r)(g))
        })
      }
      else {
        val msg1: DenseVector[Double] = !parent.tau
        val msg2 =
          if (siblings.nonEmpty) siblings.map((sibling) => !sibling.lambdaMsg).reduce(_ + _)
          else DenseVector.zeros[Double](K)
        msg1 + msg2
      }
    }

    var dist: Double = K

    val z: Param[DenseVector[Double]] = Param {
      val oldZ = z.value
      val tmp = !lambda :+ !tau
      val newZ = exp(tmp :- lse(tmp))
      dist = norm(oldZ - newZ)
      newZ
    }.default { DenseVector.rand[Double](K) }

//    def update(): Unit = {
//      val old_q = q_d.get
//      q(::, index) := q_d()
//      if (old_q != null)
//        dist = norm(old_q - q_d())
//    }

    val loglikelihood: Param[Double] = Param {
      if(parent == null)
        lse(pi.zipWithIndex.map { case (pi, g) => lse(!probW :+ log(pi * (!unode.r)(g))) })
      else
        lse(a.zipWithIndex.map { case (a, g) => lse(!probW :+ log(a.t * !parent.z * (!unode.r)(g)))})
    }
  }

  sealed class DTree(val root: DNode) {
    private def expand(node: DNode): Seq[DNode] = {
      val children =
        if (node.children != null) node.children.flatMap((child) => expand(child))
        else Seq()
      Seq(node) ++ children
    }

    lazy val nodes: Seq[DNode] = expand(root)

//    def update(): Unit = {
//      nodes.foreach { (node) => node.update() }
//    }

    def loglikelihood: Double = nodes.map(!_.loglikelihood).sum
  }

  def eStep(trees: Seq[DTree], dnodes: Seq[DNode], unodes: Seq[UNode], G: Int,
                      q: DenseMatrix[Double] = null, r: DenseMatrix[Double] = null, qr: Array[DenseMatrix[Double]] = null
                     )(implicit counters: (ParamCounter, ParamCounter)): Unit = {
    import scala.util.control.Breaks._
    val roots = dnodes.filter(_.isRoot)
    breakable {
      for(i <- 1 to 10) {
//        println(s"    - Interval $i")
//        println("      * user update")
        counters._1.update()
        unodes.par.foreach(_.r.force())
        // println(r(::, 1))
//        println("      * document update")
        counters._2.update()
        roots.par.foreach(_.z.force())
        val derror = dnodes.map((dnode) => dnode.dist).sum / dnodes.size
        val dncert = sum(dnodes.map((dnode) => if(any(!dnode.z :> 0.5)) 1 else 0))
        val uerror = unodes.map((unode) => unode.dist).sum / unodes.size
        val uncert = sum(unodes.map((unode) => if(any(!unode.r :> 0.5)) 1 else 0))
//        println(f"      $i%4d error: document $derror%4e user $uerror%4e")
//        println(f"      count: document $dncert%6d/${q.cols}%6d user $uncert%6d/${r.cols}%6d")
//        println(r(::, 0))
        if(derror <= 1e-3 && uerror <= 1e-3)
          break
      }
    }
    if(q != null)
      trees.par.foreach  { tree =>
        tree.nodes.foreach { node =>
          q(::, node.index) := !node.z
        }
      }
    if(r != null)
      unodes.par.foreach { node =>
        r(::, node.user) := !node.r
      }
    if(qr != null && q != null && r != null)
      (0 until G).foreach { (g) =>
        qr(g) := q
        unodes.foreach { (unode) =>
          unode.documents.foreach { (dnode) =>
            qr(g)(::, dnode.index) :*= (!unode.r)(g)
          }
        }
      }
  }
}