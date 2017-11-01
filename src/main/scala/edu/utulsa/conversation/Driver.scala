package edu.utulsa.conversation

import java.io.File

import edu.utulsa.cli.{param, CLIParser, validators}
import edu.utulsa.conversation.tm._
import edu.utulsa.conversation.text.Corpus

object Driver extends App {
  implicit val $: CLIParser = CLIParser.parse(args)

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
  private val action: param[String] = param("action")
    .description("Action to perform.")
    .validation(validators.IN(actions))
    .register($)

  val algorithms: Seq[String] = Seq("lda", "ntfm", "uatfm", "mmtfm")
  private val algorithm: param[String] = param("algorithm")
    .description(s"""Algorithm to use. Results are saved to a unique subdirectory.
       |
       | Choices: {${algorithms.mkString(", ")}}""".stripMargin)
    .validation(validators.IN(algorithms))
    .register($)

  val evaluators: Seq[String] = Seq("cv", "dc", "train-test")
  private val evaluator: param[String] = param("evaluator")
    .description(s"""Method for evaluating model on the dataset.
       |
       | Choices: {${algorithms.mkString(", ")}}
     """.stripMargin)
    .validation(validators.IN(evaluators))
    .default("cv")
    .register($)

  private val outputDir: param[File] = param("output-dir")
    .description("Output directory. Default: output is placed in the directory of the input file.")
    .default { new File($(corpusFile).getParent + "/" + $(algorithm) + "/") }
    .register($)

  def loadCorpus() = {
    Corpus.load($(corpusFile))
  }

  def matchAlgorithm(algorithm: String): TMAlgorithm[_] = {
    val model: TMAlgorithm[_] = algorithm match {
      case "ntfm" =>
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
    model.save($(outputDir))
  }

  println("Loading corpus...")
  val corpus = loadCorpus()

  // val (testDocs, trainDocs) = corpus.roots.splitAt(500)
  // val test = Corpus(testDocs.flatMap(corpus.expand), corpus.words, corpus.authors)
  // val train = Corpus(trainDocs.flatMap(corpus.expand), corpus.words, corpus.authors)


  $(action) match {
    case "train" =>
      val tmAlgorithm: TMAlgorithm[_] = matchAlgorithm($(algorithm))
      println("Training...")
      val model: TopicModel = tmAlgorithm.train(corpus).asInstanceOf[TopicModel]
      println("Saving to file...")
      // println(s"Left-out Likelihood: ${model.logLikelihood(test)}")
      save(model)
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
