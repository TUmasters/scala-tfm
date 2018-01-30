package edu.utulsa.conversation.tm
import java.io.File

import breeze.linalg.{Axis, DenseMatrix, DenseVector, norm, normalize}
import breeze.numerics.{exp, log}
import edu.utulsa.conversation.text.{Corpus, Document, DocumentNode}
import edu.utulsa.util.{Term, TermContainer}

class UnigramTM
(
  override val numTopics: Int,
  val numWords: Int,
  val numIterations: Int
) extends TopicModel(numTopics) with UTMParams {

  /** PARAMETERS **/
  val pi: Vector = DenseVector.rand(numTopics)
  val theta: Matrix = normalize(DenseMatrix.rand(numWords, numTopics), Axis._1, 0)

  var optim: UTMOptimizer = _

  override protected def saveModel(dir: File): Unit = ???
  override def train(corpus: Corpus): Unit = {
    optim = new UTMOptimizer(corpus, this)
    for(iter <- 1 to numIterations) {
      optim.eStep()
      optim.mStep()
    }
  }
  override def logLikelihood(corpus: Corpus): Double = {
    if(corpus == optim.corpus) optim.infer.likelihood
    else {
      val infer = new UTMInfer(corpus, this)
      infer.likelihood
    }
  }
}

sealed trait UTMParams extends TermContainer {
  val numTopics: Int
  val numWords: Int
  val pi: DenseVector[Double]
  val theta: DenseMatrix[Double]

  val logPi: Term[Vector] = Term { log(pi) }
  val logTheta: Term[Matrix] = Term { log(theta) }
}

sealed class UTMOptimizer(val corpus: Corpus, val params: UTMParams) {
  val infer = new UTMInfer(corpus, params)

  import edu.utulsa.util.math._
  import params._
  import infer._

  def mStep(): Unit = {
    pi := nodes.map(!_.q).reduce(_ + _)
    pi := normalize(pi + 1e-3, 1.0)
    theta := DenseMatrix.ones[Double](numWords, numTopics) * 1e-3
    nodes.par.foreach { node =>
      node.document.count.foreach { case (word, count) =>
          theta(word, ::) :+= (!node.q).t * count.toDouble
      }
    }
    theta := normalize(theta, Axis._0, 1.0)

    params.reset()
  }
  def eStep(): Unit = {
    infer.inferUpdate()
  }
}

sealed class UTMInfer(val corpus: Corpus, val params: UTMParams) {
  import params._
  import edu.utulsa.util.math._

  val ZERO: Vector = DenseVector.zeros(numTopics)

  def likelihood: Double = {
    inferUpdate()
    nodes.map(node => lse(!node.logPw + !node.logQ)).sum
  }

  def inferUpdate(): Unit = {
    nodes.par.foreach { n => n.reset(); n.update() }
    val dist = nodes.map(n => norm(pi - !n.q)).sum / nodes.size
    println(dist)
  }
  val nodes: Seq[DNode] = corpus.extend(new DNode(_, _))

  class DNode(override val document: Document, override val index: Int) extends DocumentNode[DNode](document, index)
    with TermContainer {
    /**
      * Per-topic word probabilities
      */
    val logPw: Term[Vector] = Term {
      document.count.map { case (word, count) =>
        (!logTheta)(word, ::).t * count.toDouble
      }.fold(ZERO)(_ + _)
    }

    /**
      * Latent topic distribution.
      */
    val logQ: Term[Vector] = Term {
      val lq = (!logPi) :+ (!logPw)
      lq - lse(lq)
    }

    val q: Term[Vector] = Term {
      exp(!logQ)
    }
  }
}