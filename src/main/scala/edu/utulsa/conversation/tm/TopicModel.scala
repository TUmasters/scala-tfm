package edu.utulsa.conversation.tm

import java.io.{File, PrintWriter}

import breeze.linalg._
import breeze.numerics.log
import edu.utulsa.cli.{Param, CLIParser, validators}
import edu.utulsa.conversation.text.{Corpus, Dictionary}

import scala.collection.mutable

case class TPair(p: Double, topic: Int)
case class DocumentTopic(id: String, topics: List[TPair])

abstract class TMAlgorithm(val numTopics: Int) {
  def train(corpus: Corpus): TopicModel
}

abstract class TopicModel
(
  val numTopics: Int,
  val corpus: Corpus,
  val documentInfo: Map[String, List[TPair]],
  val wordInfo: Map[String, List[TPair]]
) {

  def save(dir: File): Unit = {
    if(dir.exists())
      dir.delete()
    dir.mkdirs()
    saveModel(dir)
    saveData(dir)
  }

  def params: Map[String, AnyVal] = Map(
    "num-topics" -> numTopics,
    "num-documents" -> corpus.size,
    "num-words" -> corpus.words.size,
    "num-authors" -> corpus.authors.size,
    "score" -> score,
    "bic" -> bic,
    "aic" -> aic
  )

  def saveData(dir: File): Unit = {
    writeJson(new File(dir + "/document-topics.json"), documentInfo)
    writeJson(new File(dir + "/word-topics.json"), wordInfo)
    writeJson(new File(dir + "/params.json"), params)
  }

  protected def writeJson[A <: AnyRef](file: File, a: A): Unit = {
    import org.json4s._
    import org.json4s.native.Serialization
    import org.json4s.native.Serialization.writePretty

    implicit val formats = Serialization.formats(NoTypeHints)

    file.createNewFile()
    Some(new PrintWriter(file))
      .foreach { (p) => p.write(writePretty(a)); p.close() }
  }

  protected def saveModel(dir: File): Unit

  def logLikelihood(corpus: Corpus): Double

  lazy val score: Double = logLikelihood(corpus)
  lazy val bic: Double = {
    val n = corpus.documents.map(document => document.words.size).sum
    math.log(n) * numTopics + 2 * logLikelihood(corpus)
  }
  lazy val aic: Double = 2 * numTopics + 2 * logLikelihood(corpus)
}
