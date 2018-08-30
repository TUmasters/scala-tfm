package edu.utulsa.text.tm

import java.io.{File, PrintWriter}

import edu.utulsa.text.Corpus

case class TPair(p: Double, topic: Int)
case class DocumentTopic(id: String, topics: List[TPair])

abstract class TMAlgorithm(val numTopics: Int) {
  def train(corpus: Corpus): TopicModel
}

abstract class TopicModel
(
  val numTopics: Int
) {

  protected def saveModel(dir: File): Unit
  def save(dir: File): Unit = {
    if(dir.exists())
      dir.delete()
    dir.mkdirs()
    saveModel(dir)
    writeJson(new File(dir + "/params.json"), params)
  }

  def params: Map[String, AnyVal] = Map(
    "num-topics" -> numTopics
  )

  def train(corpus: Corpus): Unit

  protected def writeJson[A <: AnyRef](file: File, a: A): Unit = {
    import org.json4s._
    import org.json4s.native.Serialization
    import org.json4s.native.Serialization.writePretty

    implicit val formats: Formats = Serialization.formats(NoTypeHints)

    file.createNewFile()
    Some(new PrintWriter(file))
      .foreach { (p) => p.write(writePretty(a)); p.close() }
  }

  def logLikelihood(corpus: Corpus): Double
}
