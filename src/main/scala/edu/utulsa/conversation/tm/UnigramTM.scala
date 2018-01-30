package edu.utulsa.conversation.tm
import java.io.File

import breeze.linalg.{Axis, CSCMatrix, DenseMatrix, DenseVector, norm, normalize}
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
  val theta: Matrix = normalize(DenseMatrix.rand(numWords, numTopics), Axis._0, 1.0)

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
    nodes.map(node => lse(!node.logPw + !logPi)).sum
  }

  def approxLikelihood(numSamples: Int = 100): Double = {
    update()
    val lls = nodes.par.map(node => {
      val samples: Seq[Double] = (1 to numSamples).par.map(i => {
        // compute likelihood on this sample
        val topic = sample((!node.q).toArray.zipWithIndex.map(x => x._2 -> x._1).toMap)
        val logPZ = (!node.logPw)(topic)
        val logZ = (!logPi)(topic)
        val logQ = (!node.logQ)(topic)
        val logP = logPZ + logZ
//        if(logPZ.isInfinite || logZ.isInfinite || logQ.isInfinite) {
//          println(s" error $logPZ $logZ $logQ")
//        }
        logP - logQ
      }).seq
      lse(samples) - log(numSamples)
    })
    lls.sum
  }

  private val b: CSCMatrix[Double] = {
    val builder = new CSCMatrix.Builder[Double](corpus.size, corpus.size)

    corpus.foreach(document =>
      if (corpus.replies.contains(document))
        corpus.replies(document).foreach(reply =>
          builder.add(corpus.index(document), corpus.index(reply), 1d)
        )
    )

    builder.result()
  }

  def inferUpdate(): Unit = {
//    val q: DenseMatrix[Double] = DenseMatrix.zeros[Double](numTopics, corpus.size) // k x n
    nodes.par.foreach { n =>
      n.reset()
      n.update()
//      q(::, n.index) := !n.q
    }
//    val dist = nodes.map(n => norm(pi - !n.eDist)).sum / nodes.size
//    val a = normalize((q * b * q.t) + 1e-3, Axis._0, 1.0)
//    val p1 = nodes.map(n => lse((!n.logPw) + (!n.logQ))).sum / corpus.wordCount
//    val p2 = nodes.map { n =>
//      n.parent match {
//        case Some(p) => log((!p.q).t * a * (!n.q))
//        case None => lse((!logPi) + !n.logQ)
//      }
//    }.sum / nodes.size
//    val p3 = nodes.map(n => lse((!logPi) + !n.logQ)).sum / nodes.size
//
////    println(dist)
//    println(f"$p1%6.4f + $p2%6.4f ~ $p3%6.4f")
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

    val eDist: Term[Vector] = Term { exp((!logPw) - lse(!logPw)) }

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