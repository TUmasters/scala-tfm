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

      override def exec(): TMAlgorithm = {
        println(s"UATFM topics: ${$(numTopics)} groups: ${$(numUserGroups)} maxEIterations: ${$(numEIterations)}")
        new UATFMAlgorithm($(numTopics), $(numUserGroups), $(numIterations), $(numEIterations))
      }
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
    .add(new Command[Unit] {
      override def name = "evaluate-convs"
      override def help = "Evaluate the topic model against conversations of a particular depth."

      algorithm.register
      corpusFile.register

      val convDepth: Param[Int] = Param("depth")
        .help("Depth of conversations to use for training set.")
        .default(2)
        .register

      val resultsFile: Param[File] = Param("results-file")
        .help("File to store JSON-formatted evaluation results in.")
        .default { new File($(corpusFile).getParent + "/" + $(algorithm).name + "/results.json") }
        .register

      def split(corpus: Corpus, depth: Int): Corpus = {
        val docs: Seq[Document] = corpus.roots.flatMap(corpus.expand(_, depth = depth))
        Corpus(docs, corpus.words, corpus.authors)
      }
      override def exec(): Unit = {
        println(s"Evaluating on conversations of depth ${$(convDepth)}.")
        println("Loading corpus...")
        val corpus: Corpus = Corpus.load($(corpusFile))
        val train: Corpus = split(corpus, $(convDepth))
        val d10: Corpus = split(corpus, 10)
        val d4: Corpus = split(corpus, 4)
        val d5: Corpus = split(corpus, 5)
        val trainSize: Int = train.size
        val testSize: Int = corpus.size - train.size
        val d10Size: Int = corpus.size - d10.size

        println(s"Training ${$(algorithm).name}")
        println(s" Train size: $trainSize Test size: $testSize")
        val alg: TMAlgorithm = $(algorithm).exec()
        val model: TopicModel = alg.train(train)

        val ll1 = model.score
        val ll2 = model.logLikelihood(corpus)
        val lld10 = model.logLikelihood(d10)
        val lld4 = model.logLikelihood(d4)
        val lld5 = model.logLikelihood(d5)
        import edu.utulsa.util.writeJson
        writeJson($(resultsFile), Map(
          "depth" -> $(convDepth),
          "score" -> model.score,
          "lo-score" -> (ll2 - ll1),
          "perplexity" -> ((ll2 - ll1) / testSize),
          "d10-perplexity" -> ((ll2 - lld10) / d10Size),
          "train-size" -> trainSize,
          "test-size" -> testSize
        ))
        println(s" Perplexity: ${(ll2 - ll1) / testSize}")
        println(s" Depth 10 perplexity: ${(ll2 - lld10) / d10Size}")
        println(s" Depth 5 perplexity: ${(lld5 - lld4) / (d5.size - d4.size)}")
      }
    })
    .register

  $.parse()
  $(action).exec()
}
