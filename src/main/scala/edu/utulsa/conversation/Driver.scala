package edu.utulsa.conversation

import java.io.File

import edu.utulsa.cli.{CLIApp, CLIParser, Param, validators}
import edu.utulsa.conversation.tm._
import edu.utulsa.conversation.text.{Corpus, Document}

object Driver extends CLIApp {
  private val corpusFile: Param[File] = Param("input-file")
    .description("""JSON-formatted file of documents. Must have the following structure:
      | [ ..., {
      |  "id": <document-id>,
      |  "parent": <parent-document-id (null if no parent)>,
      |  "author": <author-id>,
      |  "words": [<word-1>, <word-2>, ..., <word-n>]
      | }, ...]""".stripMargin)
    .validation(validators.IS_FILE)
    .register

  private val actions: Seq[String] = Seq("train", "eval-depth")
  private val action: Param[String] = Param("action")
    .description("Action to perform.")
    .validation(validators.IN(actions))
    .register

  val algorithms: Seq[String] = Seq("lda", "ntfm", "uatfm", "mmtfm")
  private val algorithm: Param[String] = Param("algorithm")
    .description(s"""Algorithm to use. Results are saved to a unique subdirectory.
       |
       | Choices: {${algorithms.mkString(", ")}}""".stripMargin)
    .validation(validators.IN(algorithms))
    .register

  private val outputDir: Param[File] = Param("output-dir")
    .description("Output directory. Default: output is placed in the directory of the input file.")
    .default { new File($(corpusFile).getParent + "/" + $(algorithm) + "/") }
    .register

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

  def evalTestTrain(): Unit = {
    // val (testDocs, trainDocs) = corpus.roots.splitAt(500)
    // val test = Corpus(testDocs.flatMap(corpus.expand), corpus.words, corpus.authors)
    // val train = Corpus(trainDocs.flatMap(corpus.expand), corpus.words, corpus.authors)
  }

  def evalDepth(tm: TMAlgorithm[_]): Unit = {
    // val (testDocs, trainDocs) = corpus.roots.splitAt(100)
    // val test = Corpus(testDocs.flatMap(corpus.expand(_)), corpus.words, corpus.authors)
    // val train = Corpus(trainDocs.flatMap(corpus.expand(_)), corpus.words, corpus.authors)

    for(depth <- 3 to 10) {
      val depthDocs: Seq[Document] = corpus.roots.flatMap(corpus.expand(_, depth=depth))
      val depthCorpus: Corpus = Corpus(depthDocs, corpus.words, corpus.authors)
      val testSize: Int = corpus.size - depthCorpus.size
      println(s"""
        | DEPTH       $depth
        | # training: ${depthCorpus.size}
        | # testing:  $testSize
        | """.stripMargin)
      val model: TopicModel = tm.train(depthCorpus).asInstanceOf[TopicModel]
      val ll1 = model.score
      val ll2 = model.logLikelihood(corpus)
      // val llTest = model.logLikelihood(test)
      val ll = ll2 - ll1
      val llAvg = ll / testSize
      println(s" ll1 = $ll1")
      println(s" ll2 = $ll2")
      println(s" log-likelihood = $ll")
      println(s" avg. log-likelihood = $llAvg")
      // println(s" test log-likelihood = $llTest")
      model.save(new File($(outputDir) + s"/eval-depth/d$depth"))
    }
  }
  println("Loading corpus...")
  val corpus = loadCorpus()

  $(action) match {
    case "train" =>
      val tmAlgorithm: TMAlgorithm[_] = matchAlgorithm($(algorithm))
      println("Training...")
      val model: TopicModel = tmAlgorithm.train(corpus).asInstanceOf[TopicModel]
      println("Saving to file...")
      // println(s"Left-out Likelihood: ${model.logLikelihood(test)}")
      save(model)
    case "eval-depth" =>
      val tmAlgorithm: TMAlgorithm[_] = matchAlgorithm($(algorithm))
      evalDepth(tmAlgorithm)
  }
}
