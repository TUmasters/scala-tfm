package edu.utulsa.conversation.driver

import edu.utulsa.conversation.tm._
import org.json4s._
import org.json4s.native.JsonMethods._
import org.json4s.native.Serialization
import org.json4s.native.Serialization.write
import java.io.File
import scala.reflect.runtime.universe._

import scala.collection.mutable

case class JSONDocument(id: String, words: List[String], parent: String, author: String)

object Driver {
  private var option: Map[Argument[_], _] = Map()
  private val options = mutable.ListBuffer[Argument[_]]()
  private val positional = mutable.ArrayBuffer[Argument[_]]()
  private val optional = mutable.Map[String, Argument[_]]()
  private val positionalMatch = "(^<.+>$)".r
  private val optionalMatch = "(^--.+)$".r
  private class Argument[T: TypeTag](
    val description: String,
    val prefix: String,
    defaultValue: => Option[T] = None,
    val validate: (T) => Boolean = (arg: T) => true
  ) {
    lazy val default: Option[T] = defaultValue
    options += this
    prefix match {
      case positionalMatch(_) => positional += this
      case optionalMatch(_) => optional += prefix -> this
    }

    /**
      * Bad method of dealing with casting a string to the argument type.
      * @param value String value
      * @return Corresponding conversion to raw type
      */
    def pair(value: String): (this.type, Any) = {
      (this, typeOf[T] match {
        case t if t =:= typeOf[String] => value
        case t if t =:= typeOf[Int] => value.toInt
        case t if t =:= typeOf[Double] => value.toDouble
      })
    }
  }
  private val inputFile = new Argument[String](
    """JSON-formatted file of documents. Must have the following structure:
[ ..., {
  "id": <document-id>,
  "parent": <parent-document-id (null if no parent)>,
  "author": <author-id>,
  "words": [<word-1>, <word-2>, ..., <word-n>]
}, ...]""", "<input-file>"
  )
  private val outputDir = new Argument[String](
    """Output directory. If empty, output is placed in the directory of the input file.""",
    "--output-dir",
    { Some(new File($(inputFile)).getParent.toString) }
  )
  val algorithms: Seq[String] = Seq("lda", "ntfm", "uatfm", "mmtfm")
  private val algorithm = new Argument[String](
    """Algorithm to use. Results are saved to a unique subdirectory.""",
    "--algorithm",
    Some("mmtfm"),
    (arg: String) => algorithms contains arg
  )
  private val numTopics = new Argument[Int](
    """Number of topics to train on.""",
    "--num-topics",
    Some(10),
    (arg: Int) => arg > 0
  )
  private val numIterations = new Argument[Int](
    """Number of iterations to run algorithm for.""",
    "--num-iterations",
    Some(100),
    (arg: Int) => arg > 0
  )
  private val numUserGroups = new Argument[Int](
    """(For UATFM only) Number of user groups to train on.""",
    "--num-user-groups",
    Some(10),
    (arg: Int) => arg > 0
  )

  private def $[T](key: Argument[T]): T = {
    if(option contains key)
      option(key).asInstanceOf[T]
    else // if(option.default != None)
      key.default match {
        case Some(value) => value
        case None => throw new IllegalArgumentException(s"Argument ${key.prefix} has no value.")
      }
  }
  private def parseArguments(args: List[String], map: Map[Argument[_], _] = Map(), position: Int = 0): Map[Argument[_], _] = {
    args match {
      case Nil => map
      case optionalMatch(key) :: value :: tail =>
        parseArguments(tail, map + (optional(key) pair value), position)
      case value :: tail if position < positional.size =>
        parseArguments(tail, map + (positional(position) pair value), position + 1)
    }
  }
  lazy val usage: String = {
    (List(""" usage: java -jar tfm.jar """) :::
      positional.map((option) => s"${option.prefix} ").toList :::
      optional.values.map((option) => s"[${option.prefix} arg] ").toList
    ).mkString
  }
  lazy val help: String = {
    (List(usage, "\n\nArguments:\n") :::
      positional.map((option) => s"  ${option.prefix}\n    ${option.description}\n").toList :::
      optional.values.map((option) => s"  ${option.prefix}\n    ${option.description}\n").toList).mkString
  }

  def load(filename: String): List[JSONDocument] = {
    println("Loading JSON data...")
    implicit val formats = DefaultFormats
    val content = scala.io.Source.fromFile(filename).mkString
    val json = parse(content)
    json.extract[List[JSONDocument]]
  }

  def process(documents: Seq[JSONDocument]): Corpus = {
    println("Processing data...")
    println(" - Getting content")
    val data = documents
      .map((document) =>
        document.id -> DocumentData(document.id, document.words, document.parent, document.author))
      .toMap
    println(" - Generating corpus")
    val corpus = Corpus.build(data)
    corpus
  }

  def train(corpus: Corpus): TopicModel = {
    println("Training model...")
    val model = ($(algorithm) match {
      case _ if $(algorithm) == "lda" =>
        LDA(corpus)
      case _ if $(algorithm) == "ntfm" =>
        CTopicModel(corpus)
      case _ if $(algorithm) == "uatfm" =>
        UCTopicModel(corpus)
          .setNumUserGroups($(numUserGroups))
      case _ if $(algorithm) == "mmtfm" =>
        MMCTopicModel(corpus, $(outputDir))
    }).setK($(numTopics))
      .setNumIterations($(numIterations))
    model.train()
    model
  }

  def loadCorpus(inputFile: String) = {
    val documents = load(inputFile)
    process(documents)
  }

  def main(args: Array[String]): Unit = {
    if(args.length <= 0)
      println(usage)
    else if(args.length == 1 && (args(0) == "help" || args(0) == "--help"))
      println(help)
    else {
      option = parseArguments(args.toList)
      val corpus = loadCorpus($(inputFile))
      train(corpus)
    }
  }
}
