package edu.utulsa.conversation.tm.mtfm

import breeze.linalg._
import breeze.numerics.{exp, log}
import java.io.File

import org.json4s._
import org.json4s.native.Serialization
import org.json4s.native.Serialization.write
import java.io.PrintWriter

import edu.utulsa.cli.{CLIParser, Param}
import edu.utulsa.conversation.text.{Corpus, Dictionary, Document, DocumentNode}
import edu.utulsa.conversation.tm.MTFMParams
import edu.utulsa.util._
import edu.utulsa.util.Term

import scala.util.Random

sealed class MTFMOptimizer
(
  val corpus: Corpus,
  val params: MTFMParams
) {
  import edu.utulsa.util.math._
  import params._

  type DV = DenseVector[Double]
  type DM = DenseMatrix[Double]

  val infer = new MTFMInfer(corpus, params)
  import infer._

  val N: Int = corpus.documents.size

  val q: DM = DenseMatrix.zeros(K, N)

  val roots: Seq[DNode] = nodes.filter(_.isRoot)

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
        pi := roots.map((node) => !node.z).reduce(_ + _)
        pi := normalize(pi + 1e-3, 1.0)
      },
      () => {
        // A maximization
        a := normalize((q * b * q.t) + (1.0 / K), Axis._0, 1.0)
      },
      () => {
        // Theta maximization
        theta := DenseMatrix.ones[Double](M, K) * 0.1
        nodes.par.foreach { node =>
          node.document.count.foreach { case (word, count) =>
            theta(word, ::) :+= (!node.z).t * count.toDouble * (1-(!node.tau)(word))
          }
        }
        theta := normalize(theta, Axis._0, 1.0)
        //        theta := normalize((q * c).t + 1e-6, Axis._0, 1.0)
      },
      () => {
        rho = nodes.map(!_._tau.sumTau).sum / corpus.wordCount
        println(rho)

        phi := DenseVector.ones[Double](M) * 0.01
        nodes.par.foreach { node =>
          node.document.count.foreach { case (word, count) =>
            phi(word) :+= (!node.tau)(word) * count.toDouble
          }
        }
        phi := normalize(phi, 1.0)

        def topWords(a: DV): Unit = {
          val topWords = a.toArray.zipWithIndex.sortBy(-_._1)
            .take(10)
            .map { case (p, word) => (corpus.words(word), p) }
          println(topWords.toSeq)
        }
        topWords(phi)
        topWords(theta(::, 0))
      }
    ).par.foreach { case (step) => step() }

    params.reset()
  }
}

class MTFMInfer(val corpus: Corpus, val params: MTFMParams) {
  import edu.utulsa.util.math._
  import params._

  type DV = DenseVector[Double]
  type DM = DenseMatrix[Double]

  val N = corpus.size
  val ZERO: DV = DenseVector.zeros(K)

  val (trees, nodes) = {
    val nodes: Seq[DNode] = corpus.extend { case (d, i) => new DNode(d, i) }
    val trees: Seq[DTree] = nodes.filter(_.parent.isEmpty).map(new DTree(_))
    (trees, nodes)
  }

  def update(): Unit = {
    trees.par.foreach { case (tree) =>
      tree.nodes.foreach { node => node.reset() }
      tree.nodes.foreach { node => node.tau.update() }
      tree.nodes.foreach { node => node.z.update() }
    }
  }

  def approxLikelihood(numSamples: Int = 100): Double = {
    update()
    val lls = trees.par.map(tree => {
      val samples: Seq[Double] = (1 to numSamples).par.map(i => {
        // compute likelihood on this sample
        val topics: Map[DNode, Int] = tree.nodes.map(node => node -> node._z.draw()).toMap
        val noise: Map[DNode, Map[Int, Boolean]] = tree.nodes.map(node => node -> node._tau.draw()).toMap
        val logP = tree.nodes.map(node => node.logP(topics, noise)).sum
        val logQ = tree.nodes.map(node => node.logQ(topics, noise)).sum
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
    extends DocumentNode[DNode](document, index) {

    object _tau extends TermContainer {
      def draw(): Map[Int, Boolean] = (!tau).mapValues(Random.nextDouble() < _)
      val logTau: Term[Map[Int, Double]] = Term {
        document.count.map { case (word, _) =>
          val logP1: Double = ((!logPhi)(word) + !logRho) * K
          val logP2: Double = parent match {
            case Some(p) => log((!p.z).t * a * theta(word, ::).t) + !logRhoInv * K
            case None => sum((!logTheta)(word, ::).t + !logPi) + !logRhoInv * K
          }
          if(corpus.words(word) == "peopl") {
            println(logP1, logP2)
          }
          word -> (logP1 - lse(Seq(logP1, logP2)))
        }
      }

      val sumTau: Term[Double] = Term {
        document.count.map { case (word, count) =>
          (!tau)(word) * count
        }.sum
      }
    }
    val tau: Term[Map[Int, Double]] = Term { (!_tau.logTau).mapValues(exp(_)) }

    object _z extends TermContainer {
      def draw(): Int = sample(!topicProbs)

      /**
        * Computes log probabilities for observing a set of words for each latent
        * class.
        */
      val logPwz: Term[DV] = Term {
        document.count.map { case (word, count) =>
          (!logTheta) (word, ::).t * count.toDouble * (1-(!tau)(word))
        }.fold(ZERO)(_ + _)
      }

      val backward: Term[DenseVector[Double]] = Term {
        // Lambda messages regarding likelihood of observing the document
        val msg1 = !logPwz
        // Lambda messages from children
        var msg2 = replies.map(!_._z.backwardMsg).fold(ZERO)(_ + _)
        msg1 :+ msg2
      }

      val backwardMsg: Term[DenseVector[Double]] = Term {
        lse(a.t, !backward)
      }

      val forward: Term[DenseVector[Double]] = Term {
        parent match {
          case None => !logPi
          case Some(p) =>
            val msg1: DenseVector[Double] = !p._z.forward
            val msg2 = siblings
              .map((sibling) => !sibling._z.backwardMsg)
              .fold(ZERO)(_ + _)
            msg1 + msg2
        }
      }

      val logZ: Term[DenseVector[Double]] = Term {
        val tmp = !backward :+ !forward
        tmp - lse(tmp)
      }

      val topicProbs: Term[Map[Int, Double]] = Term {
        (!z).toArray.zipWithIndex.map(t => t._2 -> t._1).toMap
      }
    }
    val z: Term[DenseVector[Double]] = Term { exp(!_z.logZ) }.initialize(DenseVector.rand(K))

    def logP(topics: Map[DNode, Int], noises: Map[DNode, Map[Int, Boolean]]): Double = {
      val topic = topics(this)
      val noise = noises(this)
      val term1 = parent match {
        case Some(p) => (!logA)(topics(p), topic)
        case None => (!logPi)(topic)
      }
      val term2 = noise.map { case (word, isNoise) =>
        val count = document.count(word)
        if(isNoise)
          ((!logRho) + (!logPhi)(word)) * count
        else
          ((!logRhoInv) + (!logTheta)(word, topic)) * count
      }.sum
      term1 + term2
    }

    def logQ(topics: Map[DNode, Int], noises: Map[DNode, Map[Int, Boolean]]): Double = {
      val topic = topics(this)
      val noise = noises(this)
      val term1 = (!_z.logZ)(topic)
      val term2 = noise.map { case (word, isNoise) =>
        if(isNoise)
          log((!tau)(word))
        else
          log(1-(!tau)(word))
      }.sum
      term1 + term2
    }

    def reset(): Unit = {
      _tau.reset()
      _z.reset()
    }

    def update(): Unit = {
      tau.update()
      z.update()
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
  }
}
