package edu.utulsa.conversation.driver

import edu.utulsa.conversation.tm._
import org.json4s._
import org.json4s.native.JsonMethods._
import org.json4s.native.Serialization
import org.json4s.native.Serialization.{write}
import java.io.PrintWriter

case class Comment(id: String, words: List[String], parent: String, author: String)

object Reddit {
  var dataFolder = "data/reddit/politics/"
  var algorithm = "mmtfm"
  var numTopics = 50
  var numIterations = 30

  def load(filename: String): List[Comment] = {
    println("Loading JSON data...")
    implicit val formats = DefaultFormats
    val content = scala.io.Source.fromFile(filename).mkString
    val json = parse(content)
    json.extract[List[Comment]]
  }

  def process(comments: Seq[Comment]): Corpus = {
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
    }).setK(numTopics)
      .setNumIterations(numIterations)
    model.train()
    model
  }

  def results(comments: List[Comment], corpus: Corpus, model: TopicModel): Unit = {
    // val topics = (comments zip model.documentDists)
    //   .map { case (comment, document) =>
    //     DocumentData(
    //       comment.id,
    //       document.dist.take(2).map(a => TopicData(a._1, a._2)).toList
    //     ) }
    //   .toList
    // implicit val formats = Serialization.formats(NoTypeHints)
    // Some(new PrintWriter(dataFolder + "topics.json"))
    //   .foreach { p => p.write(write(topics)); p.close }
  }

  def loadCorpus(dir: String) = {
    val comments = load(dir + "comments.json")
    process(comments)
  }

  def main(args: Array[String]): Unit = {
    dataFolder = args(0)
    if(args.length > 1)
      algorithm = args(1)
    if(args.length > 2)
      numTopics = args(2).toInt
    if(args.length > 3)
      numIterations = args(3).toInt
    // val comments = load(dataFolder + "comments.json")
    // val corpus = process(comments)
    val corpus = loadCorpus(dataFolder)
    testMMCTM(corpus)
    // val model = train(corpus)
    // println("Saving results...")
    // model.save(dataFolder)
    // println("Done!")
    // results(comments, corpus, model)
  }

  def testMMCTM(corpus: Corpus): Unit = {
    val sigmaFactors: List[Double] = List(1.0) ::: (1 to 50).map((i) => 100d * (i.toDouble / 50d)).toList
    for(sigmaFactor <- sigmaFactors) {
      val folder = dataFolder + f"sigma-test/sigma-$sigmaFactor%.2f/"
      println("Training model...")
      val model = MMCTopicModel(corpus, folder)
        .setK(numTopics)
        .setNumIterations(8)
        .setSigmaFactor(sigmaFactor)
      model.train()
      model.save(folder)
      model
    }
  }
}
