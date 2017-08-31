package edu.utulsa.conversation.tm

import java.io.{File, PrintWriter}

import breeze.linalg._
import breeze.numerics.log
import edu.utulsa.conversation.text.{Corpus, Dictionary}

case class TPair(p: Double, topic: Int)
case class DocumentTopic(id: String, topics: List[TPair])

abstract class TopicModel
(
  val numTopics: Int,
  val words: Dictionary,
  val documentInfo: List[DocumentTopic]
) {

  def save(dir: File): Unit = {
    if(dir.exists())
      dir.delete()
    dir.mkdirs()
    saveModel(dir)
    saveData(dir)
  }

  def saveData(dir: File): Unit = {
    writeJson(new File(dir + "/document-topics.json"), documentInfo)
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
}
