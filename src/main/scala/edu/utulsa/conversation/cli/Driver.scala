package edu.utulsa.conversation.cli

import edu.utulsa.conversation.tm._
import java.io.File

import edu.utulsa.conversation.eval._
import edu.utulsa.conversation.text.{Corpus, DocumentData}

import scala.reflect.runtime.universe._
import scala.collection.mutable

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
    { Some(new File($(inputFile)).getParent + "/" + $(algorithm)) }
  )
  val algorithms: Seq[String] = Seq("lda", "ntfm", "uatfm", "mmtfm")
  private val algorithm = new Argument[String](
    s"""Algorithm to use. Results are saved to a unique subdirectory. Choices: $algorithms""",
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
  private val numUserGroups = new Argument[Int](
    """(For UATFM only) Number of user groups to train on.""",
    "--num-user-groups",
    Some(10),
    (arg: Int) => arg > 0
  )
  private val numIterations = new Argument[Int](
    """Number of iterations to run algorithm for.""",
    "--num-iterations",
    Some(100),
    (arg: Int) => arg > 0
  )
  private val maxEIterations = new Argument[Int](
    """Maximum number of iterations to run the E-step of the UATFM for.""",
    "--max-e-iterations",
    Some(10),
    (arg: Int) => arg > 0
  )
  private val maxAlphaIterations = new Argument[Int](
    """Maximum number of iterations to run the alpha maximization procedure for LDA.""",
    "--max-alpha-iterations",
    Some(15),
    (arg: Int) => arg > 0
  )
  val evaluators: Seq[String] = Seq("default", "cv", "dc", "train-test")
  private val evaluate = new Argument[String](
    """Method for evaluating model on the dataset.""",
    "--evaluate",
    Some("default"),
    (arg: String) => evaluators.contains(arg)
  )
  private val numFolds = new Argument[Int](
    """Number of folds for cross-validation.""",
    "--num-folds",
    Some(10),
    (arg: Int) => arg > 0
  )
  private val trainTestSplitP = new Argument[Double](
    """Train/test split proportion.""",
    "--train-test-p",
    Some(0.6),
    (arg: Double) => arg > 0 && arg < 1
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

  def loadCorpus() = {
    Corpus.load(new File($(inputFile)))
  }

  def train(corpus: Corpus): TopicModel = {
    val model: TopicModel = $(algorithm) match {
      case _ if $(algorithm) == "lda" =>
        LatentDirichletAllocation.train(corpus, $(numTopics), $(numIterations), $(maxAlphaIterations))
      case _ if $(algorithm) == "ntfm" =>
        NaiveTopicFlowModel.train(corpus, $(numTopics), $(numIterations))
      case _ if $(algorithm) == "uatfm" =>
        UserAwareTopicFlowModel.train(corpus, $(numTopics), $(numUserGroups), $(numIterations), $(maxEIterations))
      case _ if $(algorithm) == "mmtfm" =>
        MixedMembershipTopicFlowModel.train(corpus, $(numTopics), $(numIterations))
    }
      model
  }

  def save(model: TopicModel): Unit = {
    model.save(new File($(outputDir)))
  }

  def main(args: Array[String]): Unit = {
    if(args.length <= 0)
      println(usage)
    else if(args.length == 1 && (args(0) == "help" || args(0) == "--help"))
      println(help)
    else {
      option = parseArguments(args.toList)
      println("Loading corpus...")
      val corpus = loadCorpus()
      if($(evaluate) == "default") {
        println("Training model...")
        val model = train(corpus)
        println("Saving model...")
        save(model)
        println(" Saved.")
        println("Done.")
      }
      else {
        val evaluator: Evaluator = $(evaluate) match {
          case _ if $(evaluate) == "cv" => new CrossValidation($(numFolds))
          case _ if $(evaluate) == "dc" => new DiscussionCompletion()
          case _ if $(evaluate) == "train-test" => new TrainTestSplit($(trainTestSplitP))
        }
        val alg: TMAlgorithm = $(algorithm) match {
          case _ if $(algorithm) == "lda" => null
          case _ if $(algorithm) == "ntfm" => new NTFMAlgorithm()
              .setNumTopics($(numTopics))
              .setNumIterations($(numIterations))
          case _ if $(algorithm) == "uatfm" => new UATFMAlgorithm()
              .setNumTopics($(numTopics))
              .setNumUserGroups($(numUserGroups))
              .setNumIterations($(numIterations))
              .setMaxEIterations($(maxEIterations))
          case _ if $(algorithm) == "mmtfm" => null
        }
        evaluator.run(corpus, alg)
      }
    }
  }
}
