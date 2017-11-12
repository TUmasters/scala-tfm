package edu.utulsa.conversation

import java.io.File

import edu.utulsa.cli.{Action, CLIApp, Command, Param, validators}
import edu.utulsa.conversation.text.{Corpus, Document}
import edu.utulsa.conversation.tm._

object Driver extends CLIApp {
  val corpusFile: Param[File] = Param("corpus-file")
    .help("""JSON-formatted file of documents. Must have the following structure:
            | [ ..., {
            |  "id": <document-id>,
            |  "parent": <parent-document-id (null if no parent)>,
            |  "author": <author-id>,
            |  "words": [<word-1>, <word-2>, ..., <word-n>]
            | }, ...]""".stripMargin)
    .validation(validators.IS_FILE)

  val numTopics: Param[Int] = Param("num-topics")
    .help("Number of topics.")
    .default(10)
    .validation(validators.INT_GEQ(1))

  val numIterations: Param[Int] = Param("num-iterations")
    .help("Number of training iterations.")
    .default(100)
    .validation(validators.INT_GEQ(1))

  val algorithm: Action[TMAlgorithm] = Action("algorithm")
    .help("Algorithm to train corpus on.")
    .add(new Command[TMAlgorithm] {
      override val name: String = "mtfm"
      override val help: String = "Markov Topic Flow Model"

      numTopics.register
      numIterations.register

      override def exec(): TMAlgorithm = new NTFMAlgorithm($(numTopics), $(numIterations))
    })
    .add(new Command[TMAlgorithm] {
      override val name: String = "uatfm"
      override val help: String = "User-Aware Topic Flow Model"

      numTopics.register
      val numUserGroups: Param[Int] = Param("num-user-groups")
        .help("Number of user groups for the UATFM.")
        .default(10)
        .validation(validators.INT_GEQ(1))
        .register
      numIterations.register
      val numEIterations: Param[Int] = Param("num-e-iterations")
        .help("Number of iterations to run the variational inference step of the UATFM.")
        .default(10)
        .validation(validators.INT_GEQ(1))
        .register

      override def exec(): TMAlgorithm =
        new UATFMAlgorithm($(numTopics), $(numUserGroups), $(numIterations), $(numEIterations))
    })

  val action: Action[Unit] = Action("action")
    .help("Action to perform.")
    .add(new Command[Unit] {
      override val name: String = "train"
      override val help: String = "Trains a topic model on some dataset."

      algorithm.register
      corpusFile.register

      val outputDir: Param[File] = Param("output-dir")
        .help("Output directory. Default: output is placed in the directory of the input file.")
        .default { new File($(corpusFile).getParent + "/" + $(algorithm).name + "/") }
        .register

      override def exec(): Unit = {
        val alg: TMAlgorithm = $(algorithm).exec()
        println("Loading corpus...")
        val corpus: Corpus = Corpus.load($(corpusFile))
        println(s"Training ${$(algorithm).name}...")
        val model: TopicModel = alg.train(corpus)
        println(s"Saving model to '${$(outputDir)}'...")
        model.save($(outputDir))
        println("Done.")
      }
    })
    .add(new Command[Unit] {
      override val name: String = "evaluate"
      override val help: String = "Evaluates an existing model on a new dataset."

      algorithm.register
      corpusFile.register

      val testSize: Param[Int] = Param("test-size")
        .help("Number of conversations to use for the test set.")
        .default(500)
        .register

      val resultsFile: Param[File] = Param("results-file")
        .help("File to store JSON-formatted evaluation results in.")
        .default { new File($(corpusFile).getParent + "/" + $(algorithm).name + "/results.json") }
        .register

      override def exec(): Unit = {
        println("Loading corpus...")
        val corpus: Corpus = Corpus.load($(corpusFile))
        println("Generating test/train split...")
        val (testDocs: Seq[Document], trainDocs: Seq[Document]) = corpus.roots.splitAt($(testSize))
//        val test = Corpus(testDocs.flatMap(corpus.expand(_)), corpus.words, corpus.authors)
        val train = Corpus(trainDocs.flatMap(corpus.expand(_)), corpus.words, corpus.authors)
        println(s"Training ${$(algorithm).name}...")
        val alg: TMAlgorithm = $(algorithm).exec()
        val model: TopicModel = alg.train(train)
        println("Done.")
        val ll1 = model.score
        val ll2 = model.logLikelihood(corpus)
        println(s" Perplexity: $ll1")
        println(s" Left-out likelihood: ${ll2 - ll1}")
        import edu.utulsa.util.writeJson
        writeJson($(resultsFile), Map(
          "score" -> model.score,
          "lo-score" -> (ll2 - ll1),
          "test-size" -> $(testSize),
          "train-num-docs" -> train.size,
          "test-num-docs" -> (corpus.size - train.size)
        ))
      }
    })
    .register

  $.parse()
  $(action).exec()
}
