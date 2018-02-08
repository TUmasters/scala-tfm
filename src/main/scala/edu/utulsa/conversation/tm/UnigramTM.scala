package edu.utulsa.conversation.tm
import java.io.File

import breeze.linalg._
import breeze.numerics._
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

//  /** RANDOM WORD DISTRIBUTION **/
//  val rho: Vector = DenseVector(0.5)
//  val phi: Vector = normalize(DenseVector.rand(numWords))

  var optim: UTMOptimizer = _

  override protected def saveModel(dir: File): Unit = {
    require(optim != null, "Model must be trained before it can be saved.")

    import edu.utulsa.util.math.csvwritevec
    dir.mkdirs()
    println("Saving parameters...")
    csvwritevec(new File(dir + "/pi.mat"), pi)
    csvwrite(new File(dir + "/theta.mat"), theta)

    println("Saving document info...")
    val dTopics: Map[String, List[TPair]] = optim.infer.nodes.zipWithIndex.map { case (node, index) =>
      val maxItem = (!node.q).toArray.zipWithIndex
        .maxBy(_._1)
      node.document.id ->
        List(TPair(maxItem._1, maxItem._2))
    }.toMap
    writeJson(new File(dir + "/document-topics.json"), dTopics)

    println("Saving word info...")
    val wTopics: Map[String, List[TPair]] = (0 until numWords).map { case w: Int =>
      optim.corpus.words(w) ->
        theta(w, ::).t.toArray.zipWithIndex.map { case (p, i) => TPair(p, i) }.sortBy(-_.p).toList
    }.toMap
    writeJson(new File(dir + "/word-topics.json"), wTopics)
  }
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
    nodes.par.foreach { n =>
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