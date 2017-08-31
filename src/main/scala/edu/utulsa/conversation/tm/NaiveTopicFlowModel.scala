package edu.utulsa.conversation.tm

import breeze.linalg._
import breeze.numerics.{exp, log}
import java.io.File

import org.json4s._
import org.json4s.native.Serialization
import org.json4s.native.Serialization.write
import java.io.PrintWriter

import edu.utulsa.conversation.text.{Corpus, Dictionary, Document}

class NaiveTopicFlowModel
(
  override val numTopics: Int,
  override val words: Dictionary,
  override val documentInfo: List[DocumentTopic],
  val pi: DenseVector[Double],
  val a: DenseMatrix[Double],
  val theta: DenseMatrix[Double]
) extends TopicModel(numTopics, words, documentInfo) {
  override def saveModel(dir: File): Unit = {
    import MathUtils.csvwritevec
    dir.mkdirs()
    csvwritevec(new File(dir + "/pi.mat"), pi)
    csvwrite(new File(dir + "/a.mat"), a)
    csvwrite(new File(dir + "/theta.mat"), theta)
  }

  override def likelihood(corpus: Corpus): Double = ???
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
) extends TMOptimizer[NaiveTopicFlowModel](corpus, numTopics) {
  import MathUtils._

  val N = corpus.all.size
  val M = corpus.words.size

  /** MODEL PARAMETERS **/
  val pi: DenseVector[Double]          = normalize(DenseVector.rand[Double](K)) // k x 1
  val logPi: DenseVector[Double]       = log(pi)
  val a: DenseMatrix[Double]           = normalize(DenseMatrix.rand(K, K), Axis._1, 1.0) // k x k
  val logA: DenseMatrix[Double]        = log(a)
  val theta: DenseMatrix[Double]       = normalize(DenseMatrix.rand(M, K), Axis._0, 1.0) // m x k
  val logTheta: DenseMatrix[Double]    = log(theta)

  /** LATENT VARIABLE ESTIMATES **/
  private val q: DenseMatrix[Double]   = DenseMatrix.zeros[Double](K, N) // k x n

  println(" Populating responses vector...")
  /** USEFUL INTERMEDIATES **/
  private val responses: DenseVector[Double] = {
    val m = DenseVector.zeros[Double](N)
    corpus.replies.zipWithIndex.foreach{ case (document, index) =>
      m(index) = document.replies.length.toDouble
    }
    m
  }

  println(" Generating conversation matrix...")
  private val b: CSCMatrix[Double] = {
    val builder = new CSCMatrix.Builder[Double](N, N)

    corpus.replies.zipWithIndex.foreach { case (document: Document, source: Int) =>
      document.replies.foreach { case (reply: Document) =>
        builder.add(document.index, reply.index, 1d)
      }
    }

    builder.result()
  }

  println(" Generating word occurrence matrix...")
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
      println(s" Interval: $interval")
      println("  - E step")
      eStep(interval)
      println("  - M step")
      mStep(interval)
    }
    println(" Collecting results per document...")
    val d: List[DocumentTopic] = nodes.map(node => {
      val v: Array[Double] = q(::, node.index).toArray
      DocumentTopic(node.document.id, v.zipWithIndex.map { case (p, i) => TPair(p, i) }.filter(_.p > 0.1).sortBy(-_.p).toList)
    }).toList
    println(" Done.")
    new NaiveTopicFlowModel(numTopics, corpus.words, d, pi, a, theta)
  }

  val (trees: Seq[DTree], nodes: Seq[DNode]) = {
    val nodes: Seq[DNode] = corpus.map((document) =>
      new DNode(document, document.index)
    ).toSeq
    nodes.foreach { (node) =>
      val d = node.document
      d.parent match {
        case Some(parent) => node.parent = nodes(parent.index)
        case None =>
      }
      node.children = d.replies.map((c) => nodes(c.index))
    }
    val trees = nodes.filter((node) => node.parent == null)
      .map((node) => new DTree(node))
    (trees, nodes)
  }
  val roots: Seq[Int] = trees.map((tree) => tree.root.index)
  var currentTime = 1

  class Param[T >: Null]() {
    var updated: Int = 0
    var value: T = _

    def apply(): T = value

    def :=(update: => T): Unit = {
      if (currentTime > this.updated) {
        this.updated = currentTime
        this.value = update
      }
    }
  }

  sealed class DNode(val document: Document, val index: Int) {
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

    val pprobW = new Param[DenseVector[Double]]()

    /**
      * Computes log probabilities for observing a set of words for each latent
      * class.
      */
    def probW: DenseVector[Double] = {
      pprobW := {
        val result =
          if (document.words.nonEmpty)
            document.count
              .map { case (word, count) => logTheta(word, ::).t :* count.toDouble }
              .reduce(_ + _)
          else
            DenseVector.zeros[Double](K)
        result
      }
      pprobW()
    }

    val plambdaMsg = new Param[DenseVector[Double]]()

    def lambdaMsg: DenseVector[Double] = {
      plambdaMsg := {
        lse(a, this.lambda)
      }
      plambdaMsg()
    }

    val plambda = new Param[DenseVector[Double]]

    def lambda: DenseVector[Double] = {
      plambda := {
        // Lambda messages regarding likelihood of observing the document
        val msg1 = this.probW
        // Lambda messages from children
        var msg2 = DenseVector.zeros[Double](K)
        if (children.length > 0) {
          msg2 = children
            .map((child) => child.lambdaMsg)
            .reduce(_ + _)
        }
        msg1 + msg2
      }
      plambda()
    }

    val ptau = new Param[DenseVector[Double]]()

    def tau: DenseVector[Double] = {
      ptau := {
        if (parent == null) {
          logPi
        }
        else {
          val msg1: DenseVector[Double] = parent.tau
          val msg2 = siblings
            .map((sibling) => sibling.lambdaMsg)
            .fold(DenseVector.zeros[Double](K))(_ + _)
          msg1 + msg2
        }
      }
      ptau()
    }

    val pq = new Param[DenseVector[Double]]()

    def update(): Unit = {
      pq := {
        lambda + tau
      }
      val qi = exp(pq() :- lse(pq()))
      q(::, index) := qi :/ sum(qi)
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

    def update(): Unit = {
      nodes.foreach { (node) => node.update() }
    }

    def logLikelihood(): Double = {
      sum(root.pq())
    }
  }

  def eStep(interval: Int): Unit = {
    currentTime += 1
    trees.par.foreach { case (tree) => tree.update() }
  }

  def mStep(interval: Int): Unit = {
    Seq(
      () => {
        // Pi Maximization
        pi := roots.map((index) => q(::, index)).reduce(_ + _)
        pi := normalize(pi :+ 1e-4, 1.0)
        logPi := log(pi)
      },
      () => {
        // A maximization
        a := normalize((q * b * q.t) :+ 1e-4, Axis._1, 1.0)
        logA := log(a)
      },
      () => {
        // Theta maximization
        theta := normalize((q * c) :+ 1e-8, Axis._1, 1.0).t
        logTheta := log(theta)
      }
    ).par.foreach { case (step) => step() }
  }
}