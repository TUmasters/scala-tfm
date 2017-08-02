package edu.utulsa.conversation.driver

import edu.utulsa.conversation.tm._
import org.json4s._
import org.json4s.native.JsonMethods._
import org.json4s.native.Serialization
import org.json4s.native.Serialization.{write}
import java.io.PrintWriter

case class JSONDocument(id: String, words: List[String], parent: String, author: String)

object Driver {
  var dataFolder: String = null
  var algorithm = "mmtfm"
  var numTopics = 10
  var numIterations = 100

  def load(filename: String): List[JSONDocument] = {
    println("Loading JSON data...")
    implicit val formats = DefaultFormats
    val content = scala.io.Source.fromFile(filename).mkString
    val json = parse(content)
    json.extract[List[JSONDocument]]
  }

  def process(comments: Seq[JSONDocument]): Corpus = {
    println("Processing data...")
    println(" - Getting content")
    val data = comments
      .map((comment) => comment.id -> DocumentData(comment.id, comment.words, comment.parent, comment.author))
      .toMap
    println(" - Generating corpus")
    val corpus = Corpus.build(data)
    corpus
  }

  def train(corpus: Corpus): TopicModel = {
    println("Training model...")
    val model = (algorithm match {
      case _ if algorithm == "mmtfm" => MMCTopicModel(corpus, dataFolder)
      case _ if algorithm == "uatfm" => UCTopicModel(corpus).setNumUserGroups(10)
      case _ if algorithm == "ntfm" => CTopicModel(corpus)
      case _ if algorithm == "lda" => LDA(corpus)
    }).setK(numTopics)
      .setNumIterations(numIterations)
    model.train()
    model
  }

  def loadCorpus(dir: String) = {
    val comments = load(dir + "comments.json")
    process(comments)
  }

  def main(args: Array[String]): Unit = {
    require(args.length >= 1, "Must specify a data folder.")
    dataFolder = args(0)
    if(args.length > 1)
      algorithm = args(1)
    if(args.length > 2)
      numTopics = args(2).toInt
    if(args.length > 3)
      numIterations = args(3).toInt
    val corpus = loadCorpus(dataFolder)

    train(corpus)
  }
}
