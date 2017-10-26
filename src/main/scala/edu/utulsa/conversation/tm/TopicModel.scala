package edu.utulsa.conversation.tm

import java.io.{File, PrintWriter}

import breeze.linalg._
import breeze.numerics.log
import edu.utulsa.conversation.params.{Parameter, Params, validators}
import edu.utulsa.conversation.text.{Corpus, Dictionary}

import scala.collection.mutable

case class TPair(p: Double, topic: Int)
case class DocumentTopic(id: String, topics: List[TPair])

abstract class TMAlgorithm[TM <: TopicModel](implicit val $: Params) {
  protected val numTopics: Parameter[Int] = Parameter[Int](
    "num-topics",
    """The number of topics that the model will be trained on.""",
    validation = validators.INT_GEQ(1),
    default = 10
  )

  def train(corpus: Corpus): TM
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
    "num-authors" -> corpus.authors.size
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

  def likelihood(corpus: Corpus): Double

  lazy val score: Double = likelihood(corpus)
}
