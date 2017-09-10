package edu.utulsa.conversation.tm

import breeze.linalg._
import breeze.numerics.{exp, log}
import java.io.File

import org.json4s._
import org.json4s.native.Serialization
import org.json4s.native.Serialization.write
import java.io.PrintWriter

import edu.utulsa.conversation.text.{Corpus, Dictionary, Document}
import edu.utulsa.conversation.util._

sealed trait NTFMParams {
  implicit val modelParams: this.type = this
  implicit val counter: ParamCounter = new ParamCounter()
  val M: Int
  val K: Int
  val pi: DenseVector[Double]
  val logPi: Param[DenseVector[Double]] = Param {
    log(pi)
  }
  val a: DenseMatrix[Double]
  val logA: Param[DenseMatrix[Double]] = Param {
    log(a)
  }
  val theta: DenseMatrix[Double]
  val logTheta: Param[DenseMatrix[Double]] = Param {
    log(theta)
  }
}

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
    import math.csvwritevec
    dir.mkdirs()
    csvwritevec(new File(dir + "/pi.mat"), pi)
    csvwrite(new File(dir + "/a.mat"), a)
    csvwrite(new File(dir + "/theta.mat"), theta)
  }

  override def likelihood(corpus: Corpus): Double = {
    val (trees, nodes) = NTFMInfer.build(corpus)
    trees.map(_.loglikelihood).sum
  }
}

class NTFMAlgorithm extends TMAlgorithm {
  val numIterations: Parameter[Int] = new Parameter(10)
  def setNumIterations(value: Int): this.type = numIterations := value

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

class NTFMOptimizer
(
  override val corpus: Corpus,
  override val numTopics: Int,
  val numIterations: Int
) extends TMOptimizer[NaiveTopicFlowModel](corpus, numTopics) with NTFMParams {
  import math._

  val N: Int = corpus.documents.size
  val M: Int = corpus.words.size

  /** MODEL PARAMETERS **/
  val pi: DenseVector[Double]          = normalize(DenseVector.rand[Double](K)) // k x 1
  val a: DenseMatrix[Double]           = normalize(DenseMatrix.rand(K, K), Axis._1, 1.0) // k x k
  val theta: DenseMatrix[Double]       = normalize(DenseMatrix.rand(M, K), Axis._0, 1.0) // m x k

  /** LATENT VARIABLE ESTIMATES **/
  val q: DenseMatrix[Double]   = DenseMatrix.zeros[Double](K, N) // k x n

//  println(" Populating responses vector...")
//  /** USEFUL INTERMEDIATES **/
//  private val responses: DenseVector[Double] = {
//    val m = DenseVector.zeros[Double](N)
//    corpus.replies.zipWithIndex.foreach{ case (document, index) =>
//      m(index) = document.replies.length.toDouble
//    }
//    m
//  }

//  println(" Generating conversation matrix...")
  private val b: CSCMatrix[Double] = {
    val builder = new CSCMatrix.Builder[Double](N, N)

    corpus.replies.zipWithIndex.foreach { case (document: Document, source: Int) =>
      document.replies.foreach { case (reply: Document) =>
        builder.add(corpus.index(document), corpus.index(reply), 1d)
      }
    }

    builder.result()
  }

//  println(" Generating word occurrence matrix...")
  private val c: CSCMatrix[Double] = {
    val builder = new CSCMatrix.Builder[Double](N, M)

    corpus.zipWithIndex.foreach { case (document, index) =>
      document.count.foreach { case (word, count) =>
        builder.add(index, word, count.toDouble)
      }
    }

    builder.result()
  }   // n x m

  override def train(): NaiveTopicFlowModel = {
    (1 to numIterations).foreach { (interval) =>
//      println(s" Interval: $interval")
//      println("  - E step")
      eStep(interval)
//      println("  - M step")
      mStep(interval)
    }
//    println(" Collecting results per document...")
    val d: Map[String, List[TPair]] = nodes.map(node => {
      node.document.id ->
        q(::, node.index).toArray.zipWithIndex.map { case (p, i) => TPair(p, i) }.filter(_.p > 0.1).sortBy(-_.p).toList
    }).toMap
    val w: Map[String, List[TPair]] = (0 until M).map(w =>
      corpus.words(w) ->
        theta(w, ::).t.toArray.zipWithIndex.map { case (p, i) => TPair(p, i ) }.sortBy(-_.p).toList
    ).toMap
//    println(" Done.")
    new NaiveTopicFlowModel(numTopics, corpus, d, w, pi, a, theta)
  }

  val (trees, nodes) = NTFMInfer.build(corpus)
  val roots: Seq[Int] = trees.map((tree) => tree.root.index)
  var currentTime = 1

  def eStep(interval: Int): Unit = {
    counter.update()
    trees.par.foreach { case (tree) =>
      tree.nodes.foreach { node => q(::, node.index) := !node.z }
    }
  }

  def mStep(interval: Int): Unit = {
    Seq(
      () => {
        // Pi Maximization
        pi := roots.map((index) => q(::, index)).reduce(_ + _)
        pi := normalize(pi :+ 1e-4, 1.0)
      },
      () => {
        // A maximization
        a := normalize((q * b * q.t) :+ 1e-4, Axis._1, 1.0)
      },
      () => {
        // Theta maximization
        theta := normalize((q * c) :+ 1e-8, Axis._1, 1.0).t
      }
    ).par.foreach { case (step) => step() }
  }
}

object NTFMInfer {
  def build(corpus: Corpus)(implicit params: NTFMParams, counter: ParamCounter): (Seq[DTree], Seq[DNode]) = {
    val nodes: Seq[DNode] = corpus.map((document) =>
      new DNode(document, corpus.index(document))
    ).toSeq
    nodes.foreach { (node) =>
      val d = node.document
      d.parent match {
        case Some(parent) => node.parent = nodes(corpus.index(parent))
        case None =>
      }
      node.children = d.replies.map((c) => nodes(corpus.index(c)))
    }
    val trees = nodes.filter((node) => node.parent == null)
      .map((node) => new DTree(node))
    (trees, nodes)
  }

  /**
    * Inference on documents.
    */
  sealed class DNode(val document: Document, val index: Int)(implicit val params: NTFMParams) {
    import math._
    import params._
    var parent: DNode = _
    var children: Seq[DNode] = Seq()

    lazy val siblings: Seq[DNode] = {
      if (parent != null) {
        parent.children.filter((child) => child != this)
      }
      else {
        Seq()
      }
    }

    def isRoot: Boolean = this.parent == null

    /**
      * Computes log probabilities for observing a set of words for each latent
      * class.
      */
    val probW: Param[DenseVector[Double]] = Param {
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

    val lambda: Param[DenseVector[Double]] = Param {
      // Lambda messages regarding likelihood of observing the document
      val msg1 = !probW
      // Lambda messages from children
      var msg2 = DenseVector.zeros[Double](K)
      if (children.nonEmpty) {
        msg2 = children
          .map((child) => !child.lambdaMsg)
          .reduce(_ + _)
      }
      msg1 :+ msg2
    }

    val lambdaMsg: Param[DenseVector[Double]] = Param {
      lse(a, !lambda)
    }

    val tau: Param[DenseVector[Double]] = Param {
      if (parent == null) {
        !logPi
      }
      else {
        val msg1: DenseVector[Double] = !parent.tau
        val msg2 = siblings
          .map((sibling) => !sibling.lambdaMsg)
          .fold(DenseVector.zeros[Double](K))(_ + _)
        msg1 + msg2
      }
    }

    val qi: Param[DenseVector[Double]] = Param {
      !lambda :+ !tau
    }

    val z: Param[DenseVector[Double]] = Param {
      exp(!qi :- lse(!qi))
    }
      .default { normalize(DenseVector.rand[Double](K), 1.0) }

    val loglikelihood: Param[Double] = Param {
      if(parent == null)
        lse(!logPi :+ !probW)
      else
        lse(log(((!parent.z) .t * a).t) :+ !probW)
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

    //  def update(): Unit = {
    //    nodes.foreach { (node) => q(::, node.index) := !node.z }
    //  }

    def loglikelihood: Double = {
      nodes.map { node =>
        !node.loglikelihood
      }.sum
    }
  }
}
