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

class MarkovTFM
(
  override val numTopics: Int,
  val numWords: Int,
  val numIterations: Int
) extends TopicModel(numTopics) with MTFMParams {
  override val K: Int = numTopics
  override val M: Int = numWords

  override val pi: Vector =    normalize(DenseVector.rand[Double](K)) // k x 1
  override val a: Matrix =     normalize(DenseMatrix.rand(K, K), Axis._1, 1.0) // k x k
  override val theta: Matrix = normalize(DenseMatrix.rand(M, K), Axis._0, 1.0) // m x k

  override val params: Map[String, AnyVal] = super.params ++ Map(
    "num-words" -> numWords,
    "num-iterations" -> numIterations
  )
  private var optim: MTFMOptimizer = _

  override def saveModel(dir: File): Unit = {
    require(optim != null, "Model must be trained before it can be saved.")

    import edu.utulsa.util.math.csvwritevec
    dir.mkdirs()
    println("Saving parameters...")
    csvwritevec(new File(dir + "/pi.mat"), pi)
    csvwrite(new File(dir + "/a.mat"), a)
    csvwrite(new File(dir + "/theta.mat"), theta)

    println("Saving document info...")
    val dTopics: Map[String, List[TPair]] = optim.infer.nodes.zipWithIndex.map { case (node, index) =>
      val maxItem = (!node.z).toArray.zipWithIndex
        .maxBy(_._1)
      node.document.id ->
        List(TPair(maxItem._1, maxItem._2))
    }.toMap
    writeJson(new File(dir + "/document-topics.json"), dTopics)

    println("Saving word info...")
    val wTopics: Map[String, List[TPair]] = (0 until M).map { case w: Int =>
      optim.corpus.words(w) ->
        theta(w, ::).t.toArray.zipWithIndex.map { case (p, i) => TPair(p, i) }.sortBy(-_.p).toList
    }.toMap
    writeJson(new File(dir + "/word-topics.json"), wTopics)
  }

  override def train(corpus: Corpus): Unit = {
    optim = new MTFMOptimizer(corpus, this)
    optim.fit(numIterations)
  }

  override def logLikelihood(corpus: Corpus): Double = {
    if(corpus == optim.corpus) {
      optim.infer.approxLikelihood()
    }
    else {
      val infer = new MTFMInfer(corpus, this)
      infer.approxLikelihood()
    }
  }
}

sealed trait MTFMParams extends TermContainer {
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


sealed class MTFMOptimizer
(
  val corpus: Corpus,
  val params: MTFMParams
) {
  import edu.utulsa.util.math._
  import params._

  val infer = new MTFMInfer(corpus, params)
  import infer._

  val N: Int = corpus.documents.size
  /** LATENT VARIABLE ESTIMATES **/
  val q: DenseMatrix[Double] = DenseMatrix.zeros[Double](K, N) // k x n

  val roots: Seq[Int] = trees.map((tree) => tree.root.index)

  private val b: CSCMatrix[Double] = {
    val builder = new CSCMatrix.Builder[Double](N, N)

    corpus.foreach(document =>
      if (corpus.replies.contains(document))
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
  } // n x m

  def fit(numIterations: Int): Unit = {
    //    println(f"init   ${fastApproxLikelihood()}%12.4f ~ ${approxLikelihood(20)}%12.4f")
    (1 to numIterations).foreach { (interval) =>
//      println(s"Iteration $interval")
      eStep()
      //      println(f"e-step ${fastApproxLikelihood()}%12.4f ~ ${approxLikelihood(20)}%12.4f")
      mStep()
      //      println(f"m-step ${fastApproxLikelihood()}%12.4f ~ ${approxLikelihood()}%12.4f")
    }
  }

  def eStep(): Unit = {
    infer.update()
    trees.par.foreach { tree =>
      tree.nodes.foreach { node => q(::, node.index) := !node.z }
    }
  }

  def mStep(): Unit = {
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
        theta := normalize((q * c) :+ (1e-3), Axis._1, 1.0).t
      }
    ).par.foreach { case (step) => step() }

    reset()
  }
}

class MTFMInfer(val corpus: Corpus, val params: MTFMParams) {
  import edu.utulsa.util.math._
  import params._

  val N = corpus.size
  val ZERO: Vector = DenseVector.zeros(K)

  val (trees, nodes) = {
    val nodes: Seq[DNode] = corpus.extend { case (d, i) => new DNode(d, i) }
    val trees: Seq[DTree] = nodes.filter(_.parent.isEmpty).map(new DTree(_))
    (trees, nodes)
  }

  def update(): Unit = {
    trees.par.foreach { case (tree) =>
      tree.nodes.foreach { node => node.reset() }
      tree.nodes.foreach { node => node.update() }
    }
  }

  def fastApproxLikelihood(): Double = {
    nodes.par.map((node) => node.logLikelihood.get).sum
  }

  def approxLikelihood(numSamples: Int = 100): Double = {
    update()
    val lls = trees.par.map(tree => {
      val samples: Seq[Double] = (1 to numSamples).par.map(i => {
        // compute likelihood on this sample
        val topics: Map[DNode, Int] = tree.nodes.map(node => node -> sample(!node.topicProbs)).toMap
        val logPZ = tree.nodes.map(node => (!node.probW)(topics(node))).sum
        val logZ = tree.logP(topics)
        val logQ = tree.nodes.map(node => log((!node.z)(topics(node)))).sum
        val logP = logPZ + logZ
        if(logPZ.isInfinite || logZ.isInfinite || logQ.isInfinite) {
          println(s" error $logPZ $logZ $logQ")
        }
        logP - logQ
      }).seq
      lse(samples) - log(numSamples)
    })
    lls.sum
  }

  /**
    * Inference on documents.
    */
  sealed class DNode(override val document: Document, override val index: Int)
    extends DocumentNode[DNode](document, index) with TermContainer {

    /**
      * Computes log probabilities for observing a set of words for each latent
      * class.
      */
    val probW: Term[DenseVector[Double]] = Term {
      document.count.map {
        case (word, count) => (!logTheta)(word, ::).t * count.toDouble
      }
        .fold(ZERO)(_ + _)
    }

    val lambda: Term[DenseVector[Double]] = Term {
      // Lambda messages regarding likelihood of observing the document
      val msg1 = !probW
      // Lambda messages from children
      var msg2 = replies.map(!_.lambdaMsg).fold(ZERO)(_ + _)
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
            .fold(ZERO)(_ + _)
          msg1 + msg2
      }
    }

    val qi: Term[DenseVector[Double]] = Term {
      !lambda :+ !tau
    }

    val z: Term[DenseVector[Double]] = Term {
      exp(!qi :- lse(!qi))
    }.initialize { normalize(DenseVector.rand[Double](K), 1.0) }

    val topicProbs: Term[Map[Int, Double]] = Term {
      (!z).toArray.zipWithIndex.map(t => t._2 -> t._1).toMap
    }

    val logLikelihood: Term[Double] = Term {
      lse(!probW :+ !z)
    }

    val alpha1: Term[DenseVector[Double]] = Term {
      parent match {
        case Some(p) => lse(a.t, !p.alpha)
        case None => pi
      }
    }

    val alpha: Term[DenseVector[Double]] = Term {
      !probW :+ !alpha1
    }

    def logP(topics: Map[DNode, Int]): Double = {
      parent match {
        case Some(p) => (!logA)(topics(p), topics(this))
        case None => (!logPi)(topics(this))
      }
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
    lazy val leaves: Seq[DNode] = nodes.filter(_.replies.size <= 0)

    def loglikelihood: Double =
      nodes.map(_.logLikelihood.get).sum

    def logP(topics: Map[DNode, Int]): Double = nodes.map(_.logP(topics)).sum
  }
}
