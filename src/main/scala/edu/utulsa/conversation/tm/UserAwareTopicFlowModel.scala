package edu.utulsa.conversation.tm

import breeze.linalg._
import breeze.numerics.{exp, log}
import java.io.File

import edu.utulsa.conversation.text.{Corpus, Dictionary, Document, DocumentNode}
import edu.utulsa.util.{Term, TermContainer}
import edu.utulsa.util.math._

class UserAwareTopicFlowModel
(
  override val numTopics: Int,
  val numUserGroups: Int,
  override val corpus: Corpus,
  val authors: Dictionary,
  override val documentInfo: Map[String, List[TPair]],
  override val wordInfo: Map[String, List[TPair]],
  val pi: Array[DenseVector[Double]],
  val phi: DenseVector[Double],
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

  override def logLikelihood(corpus: Corpus): Double = {
    val (trees, dnodes, unodes) = UATFMInfer.build(corpus)
    UATFMInfer.update(trees, dnodes, unodes, numUserGroups, 10)
    UATFMInfer.approxLikelihood(trees, dnodes, unodes)
  }
}

trait UATFMParams extends TermContainer {
  implicit val modelParams: this.type = this
  val M: Int
  val K: Int
  val G: Int
  val pi: Array[DenseVector[Double]]
  val logPi: Term[Array[DenseVector[Double]]] = Term {
    pi.map(log(_))
  }
  val phi: DenseVector[Double]
  val logPhi: Term[DenseVector[Double]] = Term{
    log(phi)
  }
  val a: Array[DenseMatrix[Double]]
  val logA: Term[Array[DenseMatrix[Double]]] = Term {
    a.map(log(_))
  }
  val theta: DenseMatrix[Double]
  val logTheta: Term[DenseMatrix[Double]] = Term {
    log(theta)
  }
  val r: DenseMatrix[Double]
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


/**
  * UATFMOptimizer: Trains the parameters for a UATFM model on some corpus.
  * @param corpus The corpus to train on.
  * @param numTopics Number of topics to use.
  * @param numUserGroups Number of user groups to use.
  * @param numIterations Number of iterations to run for.
  * @param maxEIterations Maximum number of iterations to run for.
  */
class UATFMOptimizer
(
  val corpus: Corpus,
  val numTopics: Int,
  val numUserGroups: Int,
  val numIterations: Int,
  val maxEIterations: Int
) extends UATFMParams {

  def train(): UserAwareTopicFlowModel = {
    println(f"${0}%4d init   ${UATFMInfer.fastApproxLikelihood(trees)}%12.4f ~ ${UATFMInfer.approxLikelihood(trees, dnodes, unodes)}%12.4f")
    (1 to numIterations).foreach { (interval) =>
      eStep(interval)
      println(f"$interval%4d e-step ${UATFMInfer.fastApproxLikelihood(trees)}%12.4f ~ ${UATFMInfer.approxLikelihood(trees, dnodes, unodes)}%12.4f")
      mStep(interval)
      println(f"$interval%4d m-step ${UATFMInfer.fastApproxLikelihood(trees)}%12.4f ~ ${UATFMInfer.approxLikelihood(trees, dnodes, unodes)}%12.4f")
    }
    val d: Map[String, List[TPair]] = dnodes.map((node) =>
      node.document.id ->
        (!node.z).data.zipWithIndex.map { case (p, i) => TPair(p, i) }.sortBy(-_.p).toList
    ).toMap
    val w: Map[String, List[TPair]] = (0 until M).map((w) =>
      corpus.words(w) ->
        theta(w, ::).t.toArray.zipWithIndex.map { case (p, i) => TPair(p, i) }.sortBy(-_.p).toList
    ).toMap
    new UserAwareTopicFlowModel(numTopics, numUserGroups, corpus, corpus.authors, d, w, pi, phi, a, theta)
  }

  val N: Int = corpus.size
  val M: Int = corpus.words.size
  override val K: Int = numTopics
  val U: Int = corpus.authors.size
  val G: Int = numUserGroups

  val (trees, dnodes, unodes) = UATFMInfer.build(corpus)

  val roots: Seq[Int] = trees.map(_.root.index)

  /** MODEL PARAMETERS **/
  val pi: Array[DenseVector[Double]]    = (1 to G).map(_ => normalize(DenseVector.rand(K))).toArray // k x g
  val phi: DenseVector[Double]          = normalize(DenseVector.rand(G-1), 1.0) // 1 x g
  val a: Array[DenseMatrix[Double]]     = (1 to G).map(_ => normalize(DenseMatrix.rand(K, K), Axis._1, 1.0)).toArray // g x k x k
  val theta: DenseMatrix[Double]        = normalize(DenseMatrix.rand(M, K), Axis._0, 1.0) // m x k

  /** LATENT VARIABLE ESTIMATES **/
  val q: DenseMatrix[Double]     = normalize(DenseMatrix.rand[Double](K, N), Axis._1, 1.0) // k x n
  val r: DenseMatrix[Double]     = normalize(DenseMatrix.rand[Double](G, U), Axis._1, 1.0) // g x u
  val qr: Array[DenseMatrix[Double]] = (1 to G).map(_ => DenseMatrix.zeros[Double](K, N)).toArray // g x k x n

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
    UATFMInfer.update(trees, dnodes, unodes, G, maxEIterations)

    trees.par.foreach { tree =>
      tree.nodes.foreach { node =>
        q(::, node.index) := !node.z
      }
    }

    unodes.par.foreach { node =>
      r(::, node.user) := !node.r
    }

    (0 until G).foreach { (g) =>
      qr(g) := q
      unodes.foreach { (unode) =>
        unode.documents.foreach { (dnode) =>
          qr(g)(::, dnode.index) :*= (!unode.r) (g)
        }
      }
    }
  }

  protected def mStep(interval: Int): Unit = {
    Seq(
      () => {
        // Pi Maximization
        pi.zipWithIndex.foreach { case (pi_g, g) =>
          pi_g := normalize(roots.map((index) => qr(g)(::, index)).reduce(_ + _) :+ (1e-3/K), 1.0)
        }
      },
      () => {
        phi := normalize(unodes.map(n => (!n.r) (1 until G)).reduce(_ + _) :+ (1e-3/(G-1)), 1.0)
      },
      () => {
        // A maximization
        a.zipWithIndex.foreach { case (a_g, g) =>
          a_g := normalize(q * b * qr(g).t :+ (1e-3/(K*K)), Axis._1, 1.0)
        }
      },
      () => {
        // Theta maximization
        theta := normalize((q * c) :+ (1e-3/(K*M)), Axis._1, 1.0).t
      }
    ).par.foreach { case (step) => step() }

    reset()
  }
}


object UATFMInfer {
  def build(corpus: Corpus)(implicit params: UATFMParams): (Seq[DTree], Seq[DNode], Seq[UNode]) = {
    val dnodes: Seq[DNode] = corpus.extend[DNode] {
      case (document: Document, index: Int) => new DNode(document, index)
    }
    val authorship: Map[Int, Seq[DNode]] = dnodes.groupBy(_.document.author)
    val unodes: Seq[UNode] = corpus.authors
      .filter(authorship contains)
      .map(author => new UNode(author, authorship(author))).toSeq
    unodes.foreach { node =>
      for(dnode <- node.documents)
        dnode.author = node
    }
    val trees: Seq[DTree] = dnodes
      .filter(_.isRoot)
      .map(new DTree(_))

    (trees, dnodes, unodes)
  }

  def fastApproxLikelihood(trees: Seq[DTree]): Double = {
    trees.map(tree => tree.nodes.map(n => lse((!n.probW) :+ log(!n.z))).sum).sum
  }

  def approxLikelihood(trees: Seq[DTree], dnodes: Seq[DNode], unodes: Seq[UNode], numSamples: Int = 50): Double = {
    val samples: Seq[Double] = (1 to numSamples).par.map { case i =>
      val topics: Map[DNode, Int] = dnodes.map(node => node -> sample(!node.topicProbs)).toMap
      val groups: Map[UNode, Int] = unodes.map(node => node -> sample(!node.groupProbs)).toMap
      val logPZ: Double = dnodes.par.map(node => (!node.probW)(topics(node))).sum
      val logZ: Double = dnodes.par.map(_.logP(topics, groups)).sum +
        unodes.par.map(_.logP(topics, groups)).sum
      val logQ: Double = dnodes.par.map(node => log((!node.z)(topics(node)))).sum +
        unodes.par.map(node => log((!node.r)(groups(node)))).sum
      logPZ + logZ - logQ
    }.seq
    lse(samples) - log(samples.size)
  }

  ////////////////////////////////////////////////////////////////
  // USER NODE
  ////////////////////////////////////////////////////////////////
  sealed class UNode(val user: Int, val documents: Seq[DNode])(implicit val params: UATFMParams) extends TermContainer {

    import params._
    import edu.utulsa.util.math._

    var dist: Double = G

    val r: Term[DenseVector[Double]] = Term {
      //      val oldR = (!r).copy
      val n: DenseVector[Double] = DenseVector.zeros[Double](G)
      if(documents.length <= 2 || G <= 1) {
        n(0) = 1d
      }
      else {
        n(1 until G) := !logPhi
        n(1 until G) :+= documents.map { case (node) =>
          node.parent match {
            case None =>
              DenseVector[Double]((1 until G).map((g) =>
                log(pi(g) dot !node.z)
              ).toArray)
            case Some(parent) =>
              DenseVector[Double]((1 until G).map((g) =>
                log((!parent.z).t * a(g) * !node.z)
              ).toArray)
          }
        }.reduce(_ + _)
        n(1 until G) := exp(n(1 until G) :- lse(n(1 until G)))
        if(n(0) > 0)
          println(n)
      }
      n
    }.initialize {
      val n = DenseVector.zeros[Double](G)
      if(documents.length <= 2 || G <= 1)
        n(0) = 1d
      else
        n(1 until G) := normalize(DenseVector.rand[Double](G-1), 1.0)
      n
    }

    def logP(topics: Map[DNode, Int], groups: Map[UNode, Int]): Double = {
      if(documents.length <= 2 || G <= 1)
        1d
      else
        (!logPhi) (groups(this)-1)
    }

    val groupProbs: Term[Map[Int, Double]] = Term {
      (!r).toArray.zipWithIndex.map(t => (t._2, t._1)).toMap
    }
  }


  ////////////////////////////////////////////////////////////////
  // DOCUMENT NODE
  ////////////////////////////////////////////////////////////////
  sealed class DNode(override val document: Document, override val index: Int)(implicit val params: UATFMParams)
    extends DocumentNode[DNode](document, index) with TermContainer {

    import params._
    import edu.utulsa.util.math._

    var author: UNode = _

    /**
      * Computes log probabilities for observing a set of words for each latent
      * class.
      */
    val probW: Term[DenseVector[Double]] = Term {
      if (document.words.nonEmpty)
        document.count
          .map { case (word, count) => (!logTheta)(word, ::).t * count.toDouble }
          .reduce(_ + _)
      else
        DenseVector.zeros[Double](K)
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

    val lambdaMsg: Term[DenseVector[Double]] = Term {
      lse((0 until G).map((g) => lse(a(g), !lambda) :+ log((!author.r)(g))).toArray)
    }

    val tau: Term[DenseVector[Double]] = Term {
      implicit val tmp: UATFMParams = params
      parent match {
        case None =>
          lse((!logPi).zipWithIndex.map { case (pi_g, g) =>
            pi_g :+ log((!author.r) (g))
          })
        case Some(p) =>
          val msg1: DenseVector[Double] = !p.tau
          val msg2 = siblings
            .map((sibling) => !sibling.lambdaMsg)
            .fold(DenseVector.zeros[Double](K))(_ + _)
          msg1 + msg2
      }
    }

    var dist: Double = K

    val z: Term[DenseVector[Double]] = Term {
//      val oldZ = (!z).copy
      val tmp = !lambda :+ !tau
      val newZ = exp(tmp :- lse(tmp))
//      dist = norm(oldZ - newZ)
      newZ
    }.initialize { normalize(DenseVector.rand[Double](K), 1.0) }

    val topicProbs: Term[Map[Int, Double]] = Term {
      (!z).toArray.zipWithIndex.map(t => (t._2, t._1)).toMap
    }

    val logLikelihood: Term[Double] = Term {
      parent match {
        case None =>
          lse(pi.zipWithIndex.map { case (piG, g) => lse(!probW :+ log(piG * (!author.r)(g))) })
        case Some(parent) =>
          lse(a.zipWithIndex.map { case (aG, g) => lse(!probW :+ log(aG.t * !parent.z * (!author.r)(g)))})
      }
    }

    def logP(topics: Map[DNode, Int], groups: Map[UNode, Int]): Double = {
      val z = topics(this)
      val y = groups(author)
      val term1: Double = parent match {
        case Some(p) =>
          val zp: Int = topics(p)
          (!logA)(y)(zp, z)
        case None =>
          (!logPi)(y)(z)
      }
//      val term2: Double = replies.foldLeft(0d)((p, reply) => p + reply.logP(topics, groups))
      term1
    }
  }

  sealed class DTree(val root: DNode)(implicit val params: UATFMParams) {
    import params._
    private def expand(node: DNode): Seq[DNode] = {
      val children =
        if (node.replies != null) node.replies.flatMap((child) => expand(child))
        else Seq()
      Seq(node) ++ children
    }

    lazy val nodes: Seq[DNode] = expand(root)
    lazy val users: Seq[UNode] = nodes.map(_.author).distinct

    def reset(): Unit = nodes.foreach { (node) => node.reset() }

    def loglikelihood: Double = nodes.map(!_.logLikelihood).sum

    def logP(topics: Map[DNode, Int], groups: Map[UNode, Int]) = {
      nodes.map(_.logP(topics, groups)).sum
    }
  }

  def update(trees: Seq[DTree], dnodes: Seq[DNode], unodes: Seq[UNode], G: Int, numIterations: Int): Unit = {
    import scala.util.control.Breaks._
    val roots = dnodes.filter(_.isRoot)
    breakable {
      for(_ <- 1 to 10) {
        unodes.par.foreach(author => author.reset())
        unodes.par.foreach(author => author.r.forceUpdate())
        dnodes.par.foreach(document => document.reset())
        dnodes.par.foreach(document => document.z.forceUpdate())
        val docError: Double = dnodes.map((dnode) => dnode.dist).sum / dnodes.size
        val userError: Double = unodes.map((unode) => unode.dist).sum / unodes.size
//        val docUncert: Double = sum(dnodes.map((dnode) => if(any(!dnode.z :> 0.5)) 1 else 0))
//        val userUncert: Double = sum(unodes.map((unode) => if(any(!unode.r :> 0.5)) 1 else 0))
        if(docError <= 1e-3 && userError <= 1e-3)
          break
      }
    }
  }
}