package edu.utulsa.conversation

import java.io.File

import edu.utulsa.cli.{param, params, validators}
import edu.utulsa.conversation.tm._
import edu.utulsa.conversation.text.Corpus

object Driver extends App {
  implicit val $: params = params.parse(args)

  private val corpusFile: param[File] = param("input-file")
    .description("""JSON-formatted file of documents. Must have the following structure:
      | [ ..., {
      |  "id": <document-id>,
      |  "parent": <parent-document-id (null if no parent)>,
      |  "author": <author-id>,
      |  "words": [<word-1>, <word-2>, ..., <word-n>]
      | }, ...]""".stripMargin)
    .validation(validators.IS_FILE)
    .register($)

  private val actions: Seq[String] = Seq("train")
  private val action: param[File] = param("action")
    .description(
      """Action to perform.
      """.stripMargin)
    .validation(validators.IN(actions))

  val algorithms: Seq[String] = Seq("lda", "ntfm", "uatfm", "mmtfm")
  private val algorithm: param[String] = param("algorithm")
    .description(s"""Algorithm to use. Results are saved to a unique subdirectory.
       |
       | Choices: {${algorithms.mkString(", ")}}""".stripMargin)
    .validation(validators.IN(algorithms))

  val evaluators: Seq[String] = Seq("default", "cv", "dc", "train-test")
  private val evaluator: param[String] = param("evaluator")
    .description(s"""Method for evaluating model on the dataset.
       |
       | Choices: {${algorithms.mkString(", ")}}
     """.stripMargin)
    .validation(validators.IN(evaluators))

  private val outputDir: param[File] = param("output-dir")
    .description("Output directory. Default: output is placed in the directory of the input file.")
    .validation(validators.IS_DIRECTORY)
    .default { new File($(corpusFile).getParent + "/" + $(algorithm) + "/") }
    .register($)

  def loadCorpus() = {
    Corpus.load($(corpusFile))
  }

  def algorithm(corpus: Corpus): TMAlgorithm[_] = {
    val model: TMAlgorithm[_] = $(algorithm) match {
      case _ if $(algorithm) == "ntfm" =>
        new NTFMAlgorithm
//      case _ if $(algorithm) == "lda" =>
//        LatentDirichletAllocation.train(corpus, $(numTopics), $(numIterations), $(maxAlphaIterations))
//      case _ if $(algorithm) == "uatfm" =>
//        UserAwareTopicFlowModel.train(corpus, $(numTopics), $(numUserGroups), $(numIterations), $(maxEIterations))
//      case _ if $(algorithm) == "mmtfm" =>
//        MixedMembershipTopicFlowModel.train(corpus, $(numTopics), $(numIterations))
    }
      model
  }

  def save(model: TopicModel): Unit = {
    model.save()
  }

//  println("Loading corpus...")
//  val corpus = loadCorpus()
//  if ($(evaluator) == "default") {
//    println("Training model...")
//    val model = train(corpus)
//    println("Saving model...")
//    save(model)
//    println(" Saved.")
//    println("Done.")
//  }
//  else {
//    val evaluator: Evaluator = $(evaluator) match {
//      case _ if $(evaluator) == "cv" => new CrossValidation($(numFolds))
//      case _ if $(evaluator) == "dc" => new DiscussionCompletion()
//      case _ if $(evaluator) == "train-test" => new TrainTestSplit($(trainTestSplitP))
//    }
//    val alg: TMAlgorithm = $(algorithm) match {
//      case _ if $(algorithm) == "lda" => null
//      case _ if $(algorithm) == "ntfm" => new NTFMAlgorithm()
//        .setNumTopics($(numTopics))
//        .setNumIterations($(numIterations))
//      case _ if $(algorithm) == "uatfm" => new UATFMAlgorithm()
//        .setNumTopics($(numTopics))
//        .setNumUserGroups($(numUserGroups))
//        .setNumIterations($(numIterations))
//        .setMaxEIterations($(maxEIterations))
//      case _ if $(algorithm) == "mmtfm" => null
//    }
//    evaluator.run(corpus, alg)
//  }
}
