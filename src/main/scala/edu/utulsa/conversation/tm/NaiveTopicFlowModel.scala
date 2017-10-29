package edu.utulsa.conversation.tm

import breeze.linalg._
import breeze.numerics.{exp, log}
import java.io.File

import org.json4s._
import org.json4s.native.Serialization
import org.json4s.native.Serialization.write
import java.io.PrintWriter

import edu.utulsa.cli.{CLIParser, param}
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

  override def likelihood(corpus: Corpus): Double = {
//    // This is incorrect
////    val (trees, nodes) = NTFMInfer.build(corpus)
////    trees.map(_.loglikelihood).sum
//    val numSamples: Int = 100
//    val zy = log(normalize(theta, Axis._0, 1.0))
//    /**
//      * Sequential Importance Sampling method
//      * Based on https://www.irisa.fr/aspi/legland/ensta/ref/doucet00b.pdf
//      */
//    class SISNode(override val document: Document, override val index: Int)
//      extends DocumentNode[SISNode](document, index) {
//      // p(z|w)
//      lazy val probZW: DenseVector[Double] = {
//        if(document.words.nonEmpty)
//          document.count
//          .map { case (word, count) =>
//            zy(word, ::).t * count.toDouble
//          }.reduce(_ + _)
//        else
//          DenseVector.zeros[Double](K)
//      }
//      // p(w|z)
//      lazy val probWZ: DenseVector[Double] = {
//        if(document.words.nonEmpty)
//          document.count
//            .map { case (word, count) =>
//              //println((!logTheta)(word, ::))
//              (!logTheta)(word, ::).t * count.toDouble
//            }
//            .reduce(_ + _)
//        else
//          DenseVector.zeros[Double](K)
//      }
//      def sis(parentTopic: Option[Int] = None): (Double, Double) = {
//        import math._
//        val iDist = parentTopic match {
//          case Some(zp) =>
//            normalize(a(zp, ::) :* probZW, 1.0)
//          case None =>
//            normalize(pi :* probZW, 1.0)
//        }
//        val z: Int = sample(iDist.toArray.zipWithIndex.map { case (p, i) => (i, p)}.toMap)
//        val w: Double = probWZ(z) - log(probZW(z))
//        val f = parentTopic match {
//          case Some(zp) => log(a(zp, z))
//          case None => log(pi(z))
//        }
//        if(replies.nonEmpty) {
//          val r = replies.map(_.sis(Some(z))).unzip
//          (f + r._1.sum, w + r._2.sum)
//        }
//        else (f, w)
//      }
//    }
//    val nodes = corpus.extend(new SISNode(_, _))
//    val roots = nodes.filter(_.isRoot)
//    val score: Double = roots.map(root => {
//      val (fs, wus) = (1 until numSamples).map(_ => root.sis()).unzip
//      1d
//    }).sum
//    score
    0d
  }
}

class NTFMAlgorithm(implicit $: CLIParser) extends TMAlgorithm[NaiveTopicFlowModel] {
  val numIterations: param[Int] = param("num-iterations")
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

    corpus.foreach(document =>
      if(corpus.replies.contains(document))
        corpus.replies(document).foreach(reply =>
            builder.add(corpus.index(document), corpus.index(reply), 1d)
        )
    )
//    corpus.zipWithIndex.foreach { case (document: Document, source: Int) =>
//      document.replies.foreach { case (reply: Document) =>
//        builder.add(corpus.index(document), corpus.index(reply), 1d)
//      }
//    }

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

  def train(): NaiveTopicFlowModel = {
    (1 to numIterations).foreach { (interval) =>
      println(s"Interval $interval")
      eStep(interval)
      mStep(interval)
    }
    println("Saving document topics...")
    import edu.utulsa.util.iterable._
    val d: Map[String, List[TPair]] = nodes.zipWithIndex.map { case (node, index) =>
//      if(index % 1000 == 0) println(index)
      val maxItem = (!node.z).toArray.zipWithIndex
        .maxBy(_._1)
      node.document.id ->
        List(TPair(maxItem._1, maxItem._2))
//        (!node.z).iterator
////          .map { case (p, i) => TPair(p, i) }
////          .take(5)
//          .filter(_._2 >= 1.0 / corpus.size)
////          .topBy(10)(-_._2)
//          .toArray
//          .sortBy(-_._2)
//          .take(5)
//          .map(t => TPair(t._2, t._1)).toList
    }.toMap
    println("Saving word topics")
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

    logTheta.reset()
    logPi.reset()
    logA.reset()
  }
}

object NTFMInfer {
  def build(corpus: Corpus)(implicit params: NTFMParams): (Seq[DTree], Seq[DNode]) = {
    val nodes: Seq[DNode] = corpus.extend { case (d, i) => new DNode(d, i) }
    val trees: Seq[DTree] = nodes.filter(_.parent.isEmpty).map(new DTree(_))
    (trees, nodes)
  }

  /**
    * Inference on documents.
    */
  sealed class DNode(override val document: Document, override val index: Int)(implicit val params: NTFMParams)
    extends DocumentNode[DNode](document, index) {
    import edu.utulsa.util.math._
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

    val loglikelihood: Term[Double] = Term {
      parent match {
        case None => lse(!logPi :+ !probW)
        case Some(p) => lse(log(((!p.z) .t * a).t) :+ !probW)
      }
    }

    def reset(): Unit = {
      probW.reset()
      lambda.reset()
      lambdaMsg.reset()
      tau.reset()
      qi.reset()
      z.reset()
      loglikelihood.reset()
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

    def loglikelihood: Double = {
      nodes.map { node =>
        !node.loglikelihood
      }.sum
    }
  }
}
