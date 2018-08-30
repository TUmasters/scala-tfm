package edu.utulsa.text.tm
import java.io.File

import breeze.linalg._
import breeze.numerics._
import edu.utulsa.text.{Corpus, Document, DocumentNode}
import edu.utulsa.util.{Term, TermContainer}

class UnigramTM
(
  override val numTopics: Int,
  val numWords: Int,
  val numIterations: Int
) extends TopicModel(numTopics) with UTMParams {

  /** PARAMETERS **/
  val pi: DV = DenseVector.rand(numTopics)
  val theta: DM = normalize(DenseMatrix.rand(numWords, numTopics), Axis._0, 1.0)

  var optim: UTMOptimizer = _

  override protected def saveModel(dir: File): Unit = {
    require(optim != null, "Model must be trained before it can be saved.")

    import edu.utulsa.util.math.csvwritevec
    dir.mkdirs()
    println("Saving parameters...")
    csvwritevec(new File(dir + "/pi.mat"), pi)
    csvwrite(new File(dir + "/theta.mat"), theta)

    println("Saving document info...")
    val dTopics: Map[String, List[TPair]] = optim.infer.nodes.zipWithIndex.map { case (node, _) =>
      val maxItem = (!node.q).toArray.zipWithIndex
        .maxBy(_._1)
      node.document.id ->
        List(TPair(maxItem._1, maxItem._2))
    }.toMap
    writeJson(new File(dir + "/document-topics.json"), dTopics)

    println("Saving word info...")
    val wTopics: Map[String, List[TPair]] = (0 until numWords).map { w: Int =>
      optim.corpus.words(w) ->
        theta(w, ::).t.toArray.zipWithIndex.map { case (p, i) => TPair(p, i) }.sortBy(-_.p).toList
    }.toMap
    writeJson(new File(dir + "/word-topics.json"), wTopics)
  }
  override def train(corpus: Corpus): Unit = {
    optim = new UTMOptimizer(corpus, this)
    for(_ <- 1 to numIterations) {
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

  val logPi: Term[DV] = Term { log(pi) }
  val logTheta: Term[DM] = Term { log(theta) }
}

sealed class UTMOptimizer(val corpus: Corpus, val params: UTMParams) {
  val infer = new UTMInfer(corpus, params)

  import params._
  import infer._

  def mStep(): Unit = {
    pi := nodes.map(!_.q).reduce(_ + _)
    pi := normalize(pi + 1e-3, 1.0)
    theta := DenseMatrix.ones[Double](numWords, numTopics) * 0.1
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
  import edu.utulsa.util.math._
  import params._

  val ZERO: DV = DenseVector.zeros(numTopics)

  def likelihood: Double = {
    inferUpdate()
    nodes.map(node => lse(!node.logPw + !logPi)).sum
  }

  def approxLikelihood(numSamples: Int = 100): Double = {
    update()
    val lls = nodes.par.map(node => {
      val samples: Seq[Double] = (1 to numSamples).par.map(_ => {
        // compute likelihood on this sample
        val topic = sample((!node.q).toArray.zipWithIndex.map(x => x._2 -> x._1).toMap)
        val logPZ = (!node.logPw)(topic)
        val logZ = (!logPi)(topic)
        val logQ = (!node.logQ)(topic)
        val logP = logPZ + logZ
        logP - logQ
      }).seq
      lse(samples) - log(numSamples)
    })
    lls.sum
  }

  def inferUpdate(): Unit = {
    nodes.par.foreach { n: DNode =>
      n.reset()
      n.update()
    }
  }
  val nodes: Seq[DNode] = corpus.extend(new DNode(_, _))

  class DNode(override val document: Document, override val index: Int) extends DocumentNode[DNode](document, index)
    with TermContainer {
    /**
      * Per-topic word probabilities
      */
    val logPw: Term[DV] = Term {
      document.count.map { case (word, count) =>
        (!logTheta)(word, ::).t * count.toDouble
      }.fold(ZERO)(_ + _)
    }

    val eDist: Term[DV] = Term { exp((!logPw) - lse(!logPw)) }

    /**
      * Latent topic distribution.
      */
    val logQ: Term[DV] = Term {
      val lq = (!logPi) :+ (!logPw)
      lq - lse(lq)
    }

    val q: Term[DV] = Term {
      exp(!logQ)
    }
  }
}