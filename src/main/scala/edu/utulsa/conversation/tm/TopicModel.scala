package edu.utulsa.conversation.tm

import java.io.{File, PrintWriter}

import breeze.linalg._
import breeze.numerics.log
import edu.utulsa.conversation.text.{Corpus, Dictionary}

import scala.collection.mutable

case class TPair(p: Double, topic: Int)
case class DocumentTopic(id: String, topics: List[TPair])

abstract class TMAlgorithm {
  class Parameter[T](val default: T) {
    var value: Option[T] = None
    def :=(value: T): TMAlgorithm.this.type = {
      this.value = Some(value)
      TMAlgorithm.this
    }
  }
  def $[T](param: Parameter[T]): T = {
    param.value match {
      case Some(value) => value
      case None => param.default
    }
  }


  val numTopics: Parameter[Int] = new Parameter(10)
  def setNumTopics(value: Int): this.type = numTopics := value
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
