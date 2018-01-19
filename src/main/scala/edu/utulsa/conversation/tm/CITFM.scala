package edu.utulsa.conversation.tm

import java.io.File

import breeze.linalg._
import breeze.numerics._
import edu.utulsa.conversation.text.{Corpus, Document, DocumentNode}
import edu.utulsa.util.{Term}

import scala.util.Random

/**
  * Conversational Influence Topic Flow Model
  */

class CITFM
(
  override val numTopics: Int,
  override val numWords: Int,
  val numIterations: Int,
  override val epsilon: Double = 1e-2
) extends TopicModel(numTopics) with CITFMParams {
  override protected def saveModel(dir: File): Unit = ???
  override def logLikelihood(corpus: Corpus): Double = ???
  override def train(corpus: Corpus): Unit = {
    val inference = new CITFMInfer(corpus, this)
    val numAuthors = corpus.authors.size
    import inference._
    def eStep(): Unit = {
      // Update variational hyperparameters
//      println("  gamma")
      gamma0.update()
      gamma.update()
//      println("  zeta")
      zeta.update()
//      println("  eta")
      eta.update()
//      println("  dgGamma0")
      dgGamma0.update()
//      println("  dgGamma")
      dgGamma.update()
//      println("  dgZeta")
      dgZeta.update()
//      println("  dgEta")
      dgEta.update()

      // Update document-specific variational parameters
      // NOTE: Not doing this in any particular order right now! Might result in slowness.
      println("   psi + phi")
      for(_ <- 1 to 10) {
        nodes.par.foreach { node =>
          node.psi.update()
          node.phi.update()
        }
      }
    }
    def mAlpha(): Unit = {
      val c: DenseVector[Double] = !dgGamma0 + sum(!dgGamma, Axis._1)
      val g = (x: DenseVector[Double]) => (-digamma(x) + digamma(sum(x))) * (numTopics+1.0) + c
      val hq = (x: DenseVector[Double]) =>  trigamma(x) * (-numTopics-1.0)
      val hz = (x: DenseVector[Double]) => trigamma(sum(x)) * (numTopics + 1.0)
      alpha := newton(alpha, g, hq, hz)
    }
    def mBeta(): Unit = {
      val c: DenseVector[Double] = sum(!dgZeta, Axis._1)
      val g = (x: DenseVector[Double]) => (-digamma(x) + digamma(sum(x))) * numTopics.toDouble + c
      val hq = (x: DenseVector[Double]) => -numTopics.toDouble * trigamma(x)
      val hz = (x: DenseVector[Double]) => numTopics.toDouble * trigamma(sum(x))
      beta := newton(beta, g, hq, hz)
    }
    def mDelta(): Unit = {
      val c: DenseVector[Double] = numAuthors.toDouble * (digamma(!eta) - digamma(sum(!eta)))
      val g = (x: DenseVector[Double]) => (-digamma(x) + digamma(sum(x))) * numAuthors.toDouble + c
      val hq = (x: DenseVector[Double]) => -numAuthors.toDouble * trigamma(x)
      val hz = (x: DenseVector[Double]) => numAuthors.toDouble * trigamma(sum(x))
      delta := newton(delta, g, hq, hz)
    }
    def mStep(): Unit = {
      mAlpha()
      mBeta()
      mDelta()
    }

    for(i <- 1 to numIterations) {
      println(s"iteration $i")
      println(" e-step")
      eStep()
      println(" m-step")
      mStep()
    }
  }
}

trait CITFMParams {
  def numTopics: Int
  def numWords: Int
  def epsilon: Double
  val alpha: DenseVector[Double] = DenseVector.rand[Double](numTopics)
  val beta: DenseVector[Double] = DenseVector.rand[Double](numWords)
  val delta: DenseVector[Double] = DenseVector.rand[Double](2)

  // Constants
  val rho: DenseMatrix[Double] = (1-epsilon) * DenseMatrix.eye[Double](numTopics) +
   (epsilon / numTopics) * DenseMatrix.ones[Double](numTopics, numTopics)
  val logRho: DenseMatrix[Double] = log(rho)
}

sealed class CITFMInfer
(
  val corpus: Corpus,
  val params: CITFMParams
) {
  import params._

  val nodes: Seq[DNode] = corpus.extend(new DNode(_, _))
  val roots: Seq[DNode] = nodes.filter(_.isRoot)
  val responses: Seq[DNode] = nodes.filter(!_.isRoot)
  val words: Map[Int, Seq[DNode]] = nodes.flatMap(n => n.document.words.map(_ -> n)).groupBy(_._1).map {
    case (w, ns) => w -> ns.map(_._2)
  }

  // Variational parameters
  /**
    * Topic distribution free hyperparameters
    */
  val gamma0: Term[DenseVector[Double]] = Term {
    alpha + roots.map(!_.phi).reduce(_ + _)
  }
    .initialize(DenseVector.rand[Double](numTopics))
  val gamma: Term[DenseMatrix[Double]] = Term {
    val m = DenseMatrix.zeros[Double](numTopics, numTopics)
    m
  }
    .initialize(DenseMatrix.rand[Double](numTopics, numTopics))

  /**
    * Word distribution free hyperparameters
    */
  val zeta: Term[DenseMatrix[Double]] = Term {
    val m = tile(beta, 1, numTopics).t
    for(w <- 1 until numWords) {
      m(::, w) :+= words(w).map(!_.phi).reduce(_ + _)
    }
    m
  }.initialize(DenseMatrix.rand[Double](numTopics, numWords))

  val eta: Term[DenseVector[Double]] = Term {
    delta + responses.map(n => DenseVector(!n.psi, 1-(!n.psi))).reduce(_ + _)
  }.initialize(DenseVector.rand[Double](2))

  // These values have to be computed a lot, so let's cache them to save time.
  val dgGamma0: Term[DenseVector[Double]] = Term { digamma(!gamma0) - digamma(sum(!gamma0)) }
  val dgGamma: Term[DenseMatrix[Double]] = Term { digamma(!gamma) - tile(digamma(sum(!gamma, Axis._0)), numTopics, 1) }
  val dgZeta: Term[DenseMatrix[Double]] = Term { digamma(!zeta) - tile(digamma(sum(!zeta, Axis._0)), numTopics, 1) }
  val dgEta: Term[Double] = Term { digamma((!eta)(1)) - digamma((!eta)(2)) }

  class DNode(override val document: Document, override val index: Int) extends DocumentNode[DNode](document, index) {
    val psi: Term[Double] = Term {
      parent match {
        case Some(p) =>
          val phiMat = (!phi) * (!p.phi).t
          val term: Double = math.exp(sum(phiMat :* (logRho - !dgGamma)) + !dgEta)
          term / (term + 1)
        case None => 0d
      }
    }.initialize(Random.nextDouble())

    val phi: Term[DenseVector[Double]] = Term {
      val t1: DenseVector[Double] = document.words.map(w => (!dgZeta)(::, w)).reduce(_ + _)
      val t2: DenseVector[Double] = parent match {
        case Some(p) => sum(tile((!p.phi).t, numTopics, 1) :* (
          !psi * logRho +
            (1 - !psi) * (!dgGamma).t
        ), Axis._1)
        case None => !dgGamma0
      }
      val t3: DenseVector[Double] = replies.length match {
        case 0 => DenseVector.zeros[Double](numTopics)
        case _ => replies.map(r =>
          sum(tile((!r.phi).t, numTopics, 1) :* (
            (!r.psi) * logRho +
              (1 - !r.psi) * !dgGamma
          ), Axis._1)
        ).reduce(_ + _)
      }
      normalize(exp(t1 + t2 + t3), 1.0)
    }.initialize(normalize(DenseVector.rand[Double](numTopics), 1.0))
  }
}