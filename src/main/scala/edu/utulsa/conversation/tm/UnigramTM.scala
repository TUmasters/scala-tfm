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
  val pi: DV = DenseVector.rand(numTopics)
  val theta: DM = normalize(DenseMatrix.rand(numWords, numTopics), Axis._0, 1.0)

  /** RANDOM WORD DISTRIBUTION **/
  var rho: Double = 1e-5
  val phi: DV = normalize(DenseVector.rand(numWords), 1.0)

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
  val phi: DV

  val logPi: Term[DV] = Term { log(pi) }
  val logTheta: Term[DM] = Term { log(theta) }
  val logRho: Term[Double] = Term { log(rho) }
  val logRhoInv: Term[Double] = Term { log(1-rho) }
  val logPhi: Term[DV] = Term { log(phi) }
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
    phi := normalize(phi, 1.0)
    rho = 0.01
    params.update()

    for(iter <- 1 to numIterations) {
      eStep()
      mStep(iter)
    }
  }

  def mStep(iter: Int): Unit = {
    pi := nodes.map(!_.z).reduce(_ + _)
    pi := normalize(pi + 1e-3, 1.0)

    theta := DenseMatrix.ones[Double](numWords, numTopics) * 0.1
    nodes.par.foreach { node =>
      node.document.count.foreach { case (word, count) =>
        theta(word, ::) :+= (!node.z).t * count.toDouble * (1-(!node.tau)(word))
      }
    }
    theta := normalize(theta, Axis._0, 1.0)

    rho = nodes.map(!_._tau.sumTau).sum / nodes.map(_.numWords).sum

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

  val ZERO: DV = DenseVector.zeros(numTopics)

  def likelihood: Double = {
    inferUpdate()
    nodes.map(node => lse(!node._z.logPwz + !logPi)).sum
  }

  def approxLikelihood(numSamples: Int = 100): Double = {
    println(likelihood / corpus.wordCount)
    inferUpdate()
    val lls = nodes.par.map(node => {
      val samples: Seq[Double] = (1 to numSamples).par.map(i => {
        // compute likelihood on this sample
        val topic = node._z.draw()
        val noise = node._tau.draw()
        val logP = node.logP(topic, noise)
        val logQ = node.logQ(topic, noise)
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
    lazy val numWords: Int = document.count.values.sum

    object _tau extends TermContainer {
      def draw(): Seq[(Int, Boolean)] = {
        document.count.map { case (word, _) =>
          (word, Random.nextDouble() <= (!tau)(word))
        }.toSeq
      }

      val logTau: Term[Map[Int, Double]] = Term {
        val terms = document.count.map { case (word, count) =>
          val logP1 = ((!logPhi)(word) + !logRho) * numTopics
          val logP2 = sum((!logTheta)(word, ::).t :+ !logPi) + !logRhoInv * numTopics
//          println()
//          println(s"$logP1 $logP2")
//          println(s"${(!logPhi)(word)} ${!logRho}")
          word -> (logP1 - lse(Seq(logP1, logP2)))
        }
//        if(index == 0) println(terms)
        terms
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
    }
    val tau: Term[Map[Int, Double]] = Term { (!_tau.logTau).map(t => t._1 -> exp(t._2)) }

    /**
      * Per-topic word probabilities
      */
    object _z extends TermContainer {
      def draw(): Int = {
        sample(!topicProbs)
      }

      val logPwz: Term[DV] = Term {
        document.count.map { case (word, count) =>
          (1-(!tau)(word)) * (!logTheta)(word, ::).t * count.toDouble
        }.fold(ZERO)(_ + _)
      }

      /**
        * Latent topic distribution.
        */
      val logZ: Term[DV] = Term {
        val lq = (!logPi) :+ (!logPwz)
        lq - lse(lq)
      }.initialize(normalize(DenseVector.rand(numTopics)))

      val topicProbs: Term[Map[Int, Double]] = Term {
        (!z).toArray.zipWithIndex.map(x => x._2 -> x._1).toMap
      }
    }
    val z: Term[DV] = Term {
      exp(!_z.logZ)
    }

    def logP(topic: Int, noise: Seq[(Int, Boolean)]): Double = {
      noise.map { case (word, isNoise) =>
        val count = document.count(word)
        if(isNoise)
          ((!logRho) + (!logPhi)(word)) * count
        else
          ((!logRhoInv) + (!logTheta)(word, topic)) * count
      }.sum + (!logPi)(topic)
//      document.count.map { case (word, count) =>
//        ((!logRhoInv) + (!logTheta)(word, topic)) * count
//      }.sum + (!logPi)(topic)
    }

    def logQ(topic: Int, noise: Seq[(Int, Boolean)]): Double = {
      val term1 = (!_z.logZ)(topic)
      val term2 = noise.map { case (word, isNoise) =>
        if(isNoise)
          log((!tau)(word))
        else
          log(1-(!tau)(word))
      }.sum
      term1 + term2
    }

    def update(): Unit = {
//      for(_ <- 1 to 5) {
        _tau.reset()
        _tau.logTau.update()
        tau.reset()
        _z.reset()
        _z.logZ.update()
        z.reset()
      }
//    }
  }
}