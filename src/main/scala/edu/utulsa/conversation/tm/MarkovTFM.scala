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
import edu.utulsa.conversation.tm.mtfm.{MTFMInfer, MTFMOptimizer}
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

  override val pi: DV =    normalize(DenseVector.rand[Double](K)) // k x 1
  override val a: DM =     normalize(DenseMatrix.rand(K, K), Axis._0, 1.0) // k x k
  override val theta: DM = normalize(DenseMatrix.rand(M, K), Axis._0, 1.0) // m x k

  override var rho: Double = 0.5
  override val phi: DV   = normalize(DenseVector.rand(M), 1.0)

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

  var rho: Double
  val logRho: Term[Double] = Term { log(rho) }
  val logRhoInv: Term[Double] = Term { log(1-rho) }

  val phi: DV
  val logPhi: Term[DV] = Term { log(phi) }
}