package edu.utulsa.conversation.tm

import breeze.linalg._
import breeze.numerics.{exp, log, log1p, pow}
import java.io.File

import org.json4s._
import org.json4s.native.JsonMethods._
import org.json4s.native.Serialization
import org.json4s.native.Serialization.write
import java.io.PrintWriter

import edu.utulsa.conversation.text.{Corpus, Dictionary, Document, DocumentNode}
import edu.utulsa.util.Term
import edu.utulsa.util.math._

trait UATFMParams {
  implicit val modelParams: this.type = this
  val M: Int
  val K: Int
  val G: Int
  val pi: Array[DenseVector[Double]]
  val logPi: Term[Array[DenseVector[Double]]] = Term {
    pi.map(log(_))
  }
  val a: Array[DenseMatrix[Double]]
  val theta: DenseMatrix[Double]
  val logTheta: Term[DenseMatrix[Double]] = Term {
    log(theta)
  }
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
      edu.utulsa.util.math.csvwritevec(new File(dir + f"/pi.g$i%02d.csv"), pi(i))
      csvwrite(new File(dir + f"/a.g$i%02d.csv"), a(i))
    }
  }

  override def logLikelihood(corpus: Corpus) = ???
}

class UATFMAlgorithm
(
  override val numTopics: Int,
  val numUserGroups: Int,
  val numIterations: Int,
  val maxEIterations: Int
) extends TMAlgorithm(numTopics) {
  def train(corpus: Corpus): UserAwareTopicFlowModel = {
    new UATFMOptimizer(corpus, numTopics, numUserGroups, numIterations, maxEIterations)
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
  val corpus: Corpus,
  val numTopics: Int,
  val numUserGroups: Int,
  val numIterations: Int,
  val maxEIterations: Int
) extends UATFMParams {

  import edu.utulsa.util.math._

  def train(): UserAwareTopicFlowModel = {
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
  override val K: Int = numTopics
  val U: Int = corpus.authors.size
  val G: Int = numUserGroups

  val (trees, dnodes, unodes) = UATFMInfer.build(corpus)

  class UATFMParams
  (
    val pi: Array[DenseVector[Double]],
    val a: Array[DenseMatrix[Double]],
    val theta: DenseMatrix[Double]
  ) {
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
      m(node.index) = node.replies.length.toDouble
    }

    m
  } // n x 1
  // Reply matrix
  val b: CSCMatrix[Double] = {
    val builder = new CSCMatrix.Builder[Double](N, N)

    dnodes.foreach((node) =>
      node.replies.foreach((child) =>
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
  def build(corpus: Corpus)(implicit params: UATFMParams): (Seq[DTree], Seq[DNode], Seq[UNode]) = {
    val dnodes: Seq[DNode] = corpus.extend[DNode] {
      case (document: Document, index: Int) => new DNode(document, index)
    }
    val unodes: Seq[UNode] = corpus.authors
      .map(author => new UNode(author, dnodes.filter(_.document.author == author))).toSeq
    unodes.foreach { node =>
      for(dnode <- node.documents)
        dnode.author = node
    }
    val trees: Seq[DTree] = dnodes
      .filter(_.isRoot)
      .map(new DTree(_))

    (trees, dnodes, unodes)
  }

  ////////////////////////////////////////////////////////////////
  // USER NODE
  ////////////////////////////////////////////////////////////////
  sealed class UNode(val user: Int, val documents: Seq[DNode])(implicit val params: UATFMParams) {

    import params._
    import edu.utulsa.util.math._

    var dist: Double = G

    val r: Term[DenseVector[Double]] = Term {
      val oldR = (!r).copy
      val n: DenseVector[Double] = documents.map { case (node) =>
        node.parent match {
          case None =>
            DenseVector[Double]((0 until G).map((g) =>
              log(pi(g) dot !node.z)
            ).toArray)
          case Some(parent) =>
            DenseVector[Double]((0 until G).map((g) =>
              log((!parent.z).t * a(g) * !node.z)
            ).toArray)
        }
      }.reduce(_ + _)
      // Normalize
      val newR = exp(n :- lse(n))
      dist = norm(oldR - newR)
      newR
    }.initialize { DenseVector.rand[Double](G) }

    def reset(): Unit = {
      r.reset()
    }
  }


  ////////////////////////////////////////////////////////////////
  // DOCUMENT NODE
  ////////////////////////////////////////////////////////////////
  sealed class DNode(override val document: Document, override val index: Int)(implicit val params: UATFMParams)
    extends DocumentNode[DNode](document, index) {

    import params._
    import edu.utulsa.util.math._

    var author: UNode = _

    /**
      * Computes log probabilities for observing a set of words for each latent
      * class.
      */
    val probW: Term[DenseVector[Double]] = Term {
      val result =
        if (document.words.nonEmpty)
          document.count
            .map { case (word, count) => (!logTheta)(word, ::).t :* count.toDouble }
            .reduce(_ + _)
        else
          DenseVector.zeros[Double](K)
      result
    }

    val lambdaMsg: Term[DenseVector[Double]] = Term {
      lse((0 until G).map((g) => lse(a(g), !lambda) :+ log((!author.r)(g))).toArray)
    }

    val lambda: Term[DenseVector[Double]] = Term {
      // Lambda messages regarding likelihood of observing the document
      val msg1 = !probW
      // Lambda messages from children
      val msg2 =
        if (replies.nonEmpty)
          replies
            .map((reply) => !reply.lambdaMsg)
            .reduce(_ + _)
        else
          DenseVector.zeros[Double](K)
      msg1 + msg2
    }

    val tau: Term[DenseVector[Double]] = Term {
      implicit val tmp: UATFMParams = params
      parent match {
        case None =>
          lse((!logPi).zipWithIndex.map { case (pi_g, g) =>
            pi_g :+ log((!author.r) (g))
          })
        case Some(parent) =>
          val msg1: DenseVector[Double] = !parent.tau
          val msg2 =
            siblings.map((sibling) => !sibling.lambdaMsg).fold(DenseVector.zeros[Double](K))(_ + _)
          msg1 + msg2
      }
    }

    var dist: Double = K

    val z: Term[DenseVector[Double]] = Term {
      val oldZ = (!z).copy
      val tmp = !lambda :+ !tau
      val newZ = exp(tmp :- lse(tmp))
      dist = norm(oldZ - newZ)
      newZ
    }.initialize { DenseVector.rand[Double](K) }

    val loglikelihood: Term[Double] = Term {
      parent match {
        case None =>
          lse(pi.zipWithIndex.map { case (pi, g) => lse(!probW :+ log(pi * (!author.r)(g))) })
        case Some(parent) =>
          lse(a.zipWithIndex.map { case (a, g) => lse(!probW :+ log(a.t * !parent.z * (!author.r)(g)))})
      }
    }

    def reset(): Unit = {
      probW.reset()
      lambdaMsg.reset()
      lambda.reset()
      tau.reset()
      z.reset()
      loglikelihood.reset()
    }
  }

  sealed class DTree(val root: DNode) {
    private def expand(node: DNode): Seq[DNode] = {
      val children =
        if (node.replies != null) node.replies.flatMap((child) => expand(child))
        else Seq()
      Seq(node) ++ children
    }

    lazy val nodes: Seq[DNode] = expand(root)

    def reset(): Unit = {
      nodes.foreach { (node) => node.reset() }
    }

    def loglikelihood: Double = nodes.map(!_.loglikelihood).sum
  }

  def eStep(trees: Seq[DTree], dnodes: Seq[DNode], unodes: Seq[UNode], G: Int,
            q: DenseMatrix[Double] = null, r: DenseMatrix[Double] = null, qr: Array[DenseMatrix[Double]] = null
           ): Unit = {
    import scala.util.control.Breaks._
    val roots = dnodes.filter(_.isRoot)
    breakable {
      for(i <- 1 to 10) {
        unodes.par.foreach(author => { author.reset(); author.r.forceUpdate() })
        roots.par.foreach(document => { document.reset(); document.z.forceUpdate() })
        val derror = dnodes.map((dnode) => dnode.dist).sum / dnodes.size
        val dncert = sum(dnodes.map((dnode) => if(any(!dnode.z :> 0.5)) 1 else 0))
        val uerror = unodes.map((unode) => unode.dist).sum / unodes.size
        val uncert = sum(unodes.map((unode) => if(any(!unode.r :> 0.5)) 1 else 0))
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