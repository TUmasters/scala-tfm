package edu.utulsa.conversation.tm

import breeze.linalg._
import breeze.numerics.{exp, log}
import java.io.File

import org.json4s._
import org.json4s.native.Serialization
import org.json4s.native.Serialization.write
import java.io.PrintWriter

import edu.utulsa.cli.{CLIParser, Param}
import edu.utulsa.conversation.text.{Corpus, Dictionary, Document, DocumentNode}
import edu.utulsa.util._
import edu.utulsa.util.Term

class NaiveTopicFlowModel
(
  override val numTopics: Int,
  override val corpus: Corpus,
  override val documentInfo: Map[String, List[TPair]] = Map(),
  override val wordInfo: Map[String, List[TPair]] = Map(),
  val pi: DenseVector[Double],
  val a: DenseMatrix[Double],
  val theta: DenseMatrix[Double]
) extends TopicModel(numTopics, corpus, documentInfo, wordInfo) with NTFMParams {
  override val M: Int = corpus.words.size
  override val K: Int = numTopics

  override def saveModel(dir: File): Unit = {
    import edu.utulsa.util.math.csvwritevec
    dir.mkdirs()
    csvwritevec(new File(dir + "/pi.mat"), pi)
    csvwrite(new File(dir + "/a.mat"), a)
    csvwrite(new File(dir + "/theta.mat"), theta)
  }

  override def logLikelihood(corpus: Corpus): Double = {
    val (_, nodes) = NTFMInfer.build(corpus)
    nodes.foreach(node => node.reset())
    NTFMInfer.logLikelihood(nodes)
  }
}

class NTFMAlgorithm(implicit $: CLIParser) extends TMAlgorithm[NaiveTopicFlowModel] {
  val numIterations: Param[Int] = Param("num-iterations")
    .description("""Number of iterations to run the NTFM for.""")
    .default(10)
    .register($)

  override def train(corpus: Corpus): NaiveTopicFlowModel =
    new NTFMOptimizer(corpus, $(numTopics), $(numIterations))
      .train()
}

object NaiveTopicFlowModel {
  def train
  (
    corpus: Corpus,
    numTopics: Int,
    numIterations: Int = 10
  ): NaiveTopicFlowModel = {
    new NTFMOptimizer(corpus, numTopics, numIterations)
      .train()
  }
  def load(dir: String): NaiveTopicFlowModel = ???
}


sealed trait NTFMParams {
  implicit val modelParams: this.type = this
  val M: Int
  val K: Int
  val pi: DenseVector[Double]
  val logPi: Term[DenseVector[Double]] = Term {
    log(pi)
  }
  val a: DenseMatrix[Double]
  val logA: Term[DenseMatrix[Double]] = Term {
    log(a)
  }
  val theta: DenseMatrix[Double]
  val logTheta: Term[DenseMatrix[Double]] = Term {
    log(theta)
  }
}


sealed class NTFMOptimizer
(
  val corpus: Corpus,
  val numTopics: Int,
  val numIterations: Int
) extends NTFMParams {
  import edu.utulsa.util.math._

  val N: Int = corpus.documents.size
  val M: Int = corpus.words.size
  override val K: Int = numTopics

  /** MODEL PARAMETERS **/
  val pi: DenseVector[Double]          = normalize(DenseVector.rand[Double](K)) // k x 1
  val a: DenseMatrix[Double]           = normalize(DenseMatrix.rand(K, K), Axis._1, 1.0) // k x k
  val theta: DenseMatrix[Double]       = normalize(DenseMatrix.rand(M, K), Axis._0, 1.0) // m x k

  /** LATENT VARIABLE ESTIMATES **/
  val q: DenseMatrix[Double]   = DenseMatrix.zeros[Double](K, N) // k x n

  private val b: CSCMatrix[Double] = {
    val builder = new CSCMatrix.Builder[Double](N, N)

    corpus.foreach(document =>
      if(corpus.replies.contains(document))
        corpus.replies(document).foreach(reply =>
            builder.add(corpus.index(document), corpus.index(reply), 1d)
        )
    )

    builder.result()
  }

  private val c: CSCMatrix[Double] = {
    val builder = new CSCMatrix.Builder[Double](N, M)

    corpus.zipWithIndex.foreach { case (document, index) =>
      document.count.foreach { case (word, count) =>
        builder.add(index, word, count.toDouble)
      }
    }

    builder.result()
  }   // n x m

  def train(): NaiveTopicFlowModel = {
    (1 to numIterations).foreach { (interval) =>
      eStep(interval)
      mStep(interval)
    }
    val d: Map[String, List[TPair]] = nodes.zipWithIndex.map { case (node, index) =>
      val maxItem = (!node.z).toArray.zipWithIndex
        .maxBy(_._1)
      node.document.id ->
        List(TPair(maxItem._1, maxItem._2))
    }.toMap
    val w: Map[String, List[TPair]] = (0 until M).map(w =>
      corpus.words(w) ->
        theta(w, ::).t.toArray.zipWithIndex.map { case (p, i) => TPair(p, i ) }.sortBy(-_.p).toList
    ).toMap
    new NaiveTopicFlowModel(numTopics, corpus, d, w, pi, a, theta)
  }

  val (trees, nodes) = NTFMInfer.build(corpus)
  val roots: Seq[Int] = trees.map((tree) => tree.root.index)
  var currentTime = 1

  def eStep(interval: Int): Unit = {
    trees.par.foreach { case (tree) =>
      tree.nodes.foreach { node => node.reset() }
      tree.nodes.foreach { node => q(::, node.index) := !node.z }
    }
  }

  def mStep(interval: Int): Unit = {
    Seq(
      () => {
        // Pi Maximization
        pi := roots.map((index) => q(::, index)).reduce(_ + _)
        pi := normalize(pi :+ (1e-3 / K), 1.0)
      },
      () => {
        // A maximization
        a := normalize((q * b * q.t) :+ (1e-3 / K), Axis._1, 1.0)
      },
      () => {
        // Theta maximization
        theta := normalize((q * c) :+ (1e-7 / K), Axis._1, 1.0).t
      }
    ).par.foreach { case (step) => step() }

    logTheta.reset()
    logPi.reset()
    logA.reset()
  }
}

object NTFMInfer {
  import edu.utulsa.util.math._

  def build(corpus: Corpus)(implicit params: NTFMParams): (Seq[DTree], Seq[DNode]) = {
    val nodes: Seq[DNode] = corpus.extend { case (d, i) => new DNode(d, i) }
    val trees: Seq[DTree] = nodes.filter(_.parent.isEmpty).map(new DTree(_))
    (trees, nodes)
  }

  def logLikelihood(nodes: Seq[DNode]): Double =
    nodes.par.map((node) => node.logLikelihood.get).sum

  /**
    * Inference on documents.
    */
  sealed class DNode(override val document: Document, override val index: Int)(implicit val params: NTFMParams)
    extends DocumentNode[DNode](document, index) {
    import params._

    lazy val siblings: Seq[DNode] = {
      parent match {
        case None => Seq()
        case Some(p) => p.replies.filter(child => child != this)
      }
    }

    /**
      * Computes log probabilities for observing a set of words for each latent
      * class.
      */
    val probW: Term[DenseVector[Double]] = Term {
      if(document.words.nonEmpty)
        document.count
          .map { case (word, count) =>
            //println((!logTheta)(word, ::))
            (!logTheta)(word, ::).t * count.toDouble
          }
          .reduce(_ + _)
      else
        DenseVector.zeros[Double](K)
    }

    val lambda: Term[DenseVector[Double]] = Term {
      // Lambda messages regarding likelihood of observing the document
      val msg1 = !probW
      // Lambda messages from children
      var msg2 = DenseVector.zeros[Double](K)
      if (replies.nonEmpty) {
        msg2 = replies
          .map((child) => !child.lambdaMsg)
          .reduce(_ + _)
      }
      msg1 :+ msg2
    }

    val lambdaMsg: Term[DenseVector[Double]] = Term {
      lse(a, !lambda)
    }

    val tau: Term[DenseVector[Double]] = Term {
      parent match {
        case None => !logPi
        case Some(p) =>
          val msg1: DenseVector[Double] = !p.tau
          val msg2 = siblings
            .map((sibling) => !sibling.lambdaMsg)
            .fold(DenseVector.zeros[Double](K))(_ + _)
          msg1 + msg2
      }
    }

    val qi: Term[DenseVector[Double]] = Term {
      !lambda :+ !tau
    }

    val z: Term[DenseVector[Double]] = Term {
      exp(!qi :- lse(!qi))
    }
      .initialize { normalize(DenseVector.rand[Double](K), 1.0) }

    val logLikelihood: Term[Double] = Term {
      lse(!probW :+ !z)
    }

    def reset(): Unit = {
      probW.reset()
      lambda.reset()
      lambdaMsg.reset()
      tau.reset()
      qi.reset()
      z.reset()
      logLikelihood.reset()
    }
  }

  sealed class DTree(val root: DNode) {
    private def expand(node: DNode): Seq[DNode] = {
      val children =
        if (node.replies.nonEmpty) node.replies.flatMap((child) => expand(child))
        else Seq()
      Seq(node) ++ children
    }

    lazy val nodes: Seq[DNode] = expand(root)

    def loglikelihood: Double =
      nodes.map(_.logLikelihood.get).sum
  }
}
