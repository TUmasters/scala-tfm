package edu.utulsa.conversation.tm
import java.io.File

import breeze.linalg._
import breeze.numerics._
import edu.utulsa.conversation.text.{Corpus, Document, DocumentNode}
import edu.utulsa.util.{Term, TermContainer}

import scala.util.Random

class UnigramTM
(
  override val numTopics: Int,
  val numWords: Int,
  val numIterations: Int
) extends TopicModel(numTopics) with UTMParams {

  /** PARAMETERS **/
  val pi: Vector = DenseVector.rand(numTopics)
  val theta: Matrix = normalize(DenseMatrix.rand(numWords, numTopics), Axis._0, 1.0)

  /** RANDOM WORD DISTRIBUTION **/
  var rho: Double = 0.1
  val phi: Vector = normalize(DenseVector.rand(numWords), 1.0)

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
      val maxItem = (!node.z).toArray.zipWithIndex
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
    optim.train(numIterations)
  }
  override def logLikelihood(corpus: Corpus): Double = {
    if(corpus == optim.corpus) optim.infer.approxLikelihood()
    else {
      val infer = new UTMInfer(corpus, this)
      infer.approxLikelihood()
    }
  }
}

sealed trait UTMParams extends TermContainer {
  val numTopics: Int
  val numWords: Int
  val pi: DenseVector[Double]
  val theta: DenseMatrix[Double]
  var rho: Double
  val phi: Vector

  val logPi: Term[Vector] = Term { log(pi) }
  val logTheta: Term[Matrix] = Term { log(theta) }
  val logRho: Term[Double] = Term { log(rho) }
  val logRhoInv: Term[Double] = Term { log(1-rho) }
  val logPhi: Term[Vector] = Term { log(phi) }
}

sealed class UTMOptimizer(val corpus: Corpus, val params: UTMParams) {
  val infer = new UTMInfer(corpus, params)

  import edu.utulsa.util.math._
  import params._
  import infer._

  def train(numIterations: Int): Unit = {
    for(document <- corpus.documents) {
      document.count.foreach { case (word, count) =>
        phi(word) :+= count.toDouble
      }
    }
    rho = 1e-6
    params.update()

    for(iter <- 1 to numIterations) {
      eStep()
      mStep()
    }
  }

  def mStep(): Unit = {
    pi := nodes.map(!_.z).reduce(_ + _)
    pi := normalize(pi + 1e-3, 1.0)

    theta := DenseMatrix.ones[Double](numWords, numTopics) * 0.1
    nodes.par.foreach { node =>
      node.document.count.foreach { case (word, count) =>
        theta(word, ::) :+= (!node.z).t * count.toDouble * (1-(!node.tau)(word))
      }
    }
    theta := normalize(theta, Axis._0, 1.0)

    rho = nodes.map(!_.tau_terms.sumTau).sum / nodes.map(_.numWords).sum
    println(rho)

    phi := DenseVector.ones[Double](numWords) * 0.1
    nodes.par.foreach { node =>
      node.document.count.foreach { case (word, count) =>
        phi(word) :+= (!node.tau)(word) * count.toDouble
      }
    }
    phi := normalize(phi, 1.0)

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

//  def likelihood: Double = {
//    inferUpdate()
//    nodes.map(node => lse(!node.logPwz + !logPi)).sum
//  }

  def approxLikelihood(numSamples: Int = 100): Double = {
    inferUpdate()
    val rand = new Random()
    val lls = nodes.par.map(node => {
      val samples: Seq[Double] = (1 to numSamples).par.map(i => {
        // compute likelihood on this sample
        val topic = sample((!node.z).toArray.zipWithIndex.map(x => x._2 -> x._1).toMap)
        val noise = node.document.words.map { word => word -> (rand.nextDouble() < (!node.tau)(word)) }.toMap
        val logP = node.logP(topic, noise)
        val logZ = (!logPi)(topic)
        val logQ = (!node.z_terms.logZ)(topic) + node.tau_terms.logP(noise)
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
      n.update()
    }
  }
  val nodes: Seq[DNode] = corpus.extend(new DNode(_, _))

  class DNode(override val document: Document, override val index: Int) extends DocumentNode[DNode](document, index) {
    lazy val numWords: Int = document.count.map(_._2).sum

    object tau_terms extends TermContainer {
      val logTau: Term[Map[Int, Double]] = Term {
        document.count.map { case (word, count) =>
          val logP1 = (!logPhi)(word) + !logRho
          val logP2 = sum((!logTheta)(word, ::).t :* !z) + !logRhoInv
          word -> (logP1 - lse(Seq(logP1, logP2)))
        }.toMap
      }.initialize {
        document.words.map { word =>
          word -> -100d
        }.toMap
      }
      val sumTau: Term[Double] = Term {
        document.count.map { case (word, count) =>
          (!tau)(word) * count
        }.sum
      }
      def logP(noise: Map[Int, Boolean]): Double = {
        document.words.map { word =>
          if(noise(word))
            (!logTau)(word)
          else
            1-(!logTau)(word)
        }.sum
      }
    }
    val tau: Term[Map[Int, Double]] = Term { (!tau_terms.logTau).map(t => t._1 -> exp(t._2)) }

    /**
      * Per-topic word probabilities
      */
    object z_terms extends TermContainer {
      val logPwz: Term[Vector] = Term {
        document.count.map { case (word, count) =>
          (1-(!tau)(word)) * (!logTheta)(word, ::).t * count.toDouble
        }.fold(ZERO)(_ + _)
      }

      /**
        * Latent topic distribution.
        */
      val logZ: Term[Vector] = Term {
        val lq = (!logPi) :+ (!logPwz)
        lq - lse(lq)
      }.initialize(normalize(DenseVector.rand(numTopics)))
    }
    val z: Term[Vector] = Term {
      exp(!z_terms.logZ)
    }

    def logP(topic: Int, noise: Map[Int, Boolean]): Double = {
      document.count.map { case (word, count) =>
        if(noise(word))
          ((!logRho) + (!logPhi)(word)) * count
        else
          ((!logRhoInv) + (!logTheta)(word, topic)) * count
      }.sum + (!logPi)(topic)
    }

    def update(): Unit = {
      for(_ <- 1 to 5) {
        tau_terms.reset()
        tau_terms.logTau.update()
        tau.reset()
        z_terms.reset()
        z_terms.logZ.update()
        z.reset()
      }
    }
  }
}