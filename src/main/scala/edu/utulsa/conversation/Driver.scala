package edu.utulsa.conversation

import java.io.File

import edu.utulsa.cli.{Action, CLIApp, Command, Param, ParamConverter, validators}
import edu.utulsa.conversation.text.{Corpus, Document}
import edu.utulsa.conversation.tm._

object Driver extends CLIApp {
  implicit object CorpusConverter extends ParamConverter[Corpus] {
    override def decode(filename: String): Corpus = {
      val corpusFile = new File(filename)
      Corpus.load(corpusFile)
    }
  }

  val corpus: Param[Corpus] = Param("corpus")
    .help("""JSON-formatted file of documents. Must have the following structure:
            | [ ..., {
            |  "id": <document-id>,
            |  "parent": <parent-document-id (null if no parent)>,
            |  "author": <author-id>,
            |  "words": [<word-1>, <word-2>, ..., <word-n>]
            | }, ...]""".stripMargin)


  val numTopics: Param[Int] = Param("num-topics")
    .help("Number of topics.")
    .default(10)
    .validation(validators.INT_GEQ(1))

  val numIterations: Param[Int] = Param("num-iterations")
    .help("Number of training iterations.")
    .default(100)
    .validation(validators.INT_GEQ(1))

  val algorithm: Action[TopicModel] = Action("algorithm")
    .help("Algorithm to train corpus on.")
    .add(new Command[TopicModel] {
      override def name: String = "citfm"
      override def help: String = "Conversational Influence Topic Flow Model"

      numTopics.register
      numIterations.register

      override def exec() = new CITFM($(numTopics), $(corpus).words.size, $(numIterations))
    })
    .add(new Command[TopicModel] {
      override val name: String = "mtfm"
      override val help: String = "Markov Topic Flow Model"

      numTopics.register
      numIterations.register

      override def exec(): TopicModel = new MarkovTFM($(numTopics), $(corpus).words.size, $(numIterations))
    })
    .add(new Command[TopicModel] {
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

      override def exec(): TopicModel = {
        println(s"UATFM topics: ${$(numTopics)} groups: ${$(numUserGroups)} maxEIterations: ${$(numEIterations)}")
        new AuthorAwareTFM($(numTopics), $(corpus).words.size, $(numUserGroups), $(numIterations), $(numEIterations))
      }
    })

  val action: Action[Unit] = Action("action")
    .help("Action to perform.")
    .add(new Command[Unit] {
      override val name: String = "train"
      override val help: String = "Trains a topic model on some dataset."

      algorithm.register
      corpus.register

      val outputDir: Param[File] = Param("output-dir")
        .help("Output directory. Default: output is placed in the directory of the input file.")
        .default { new File($(corpus).path.getParent + "/" + $(algorithm).name + "/") }
        .register

      override def exec(): Unit = {
        val model = $(algorithm).exec()
        println(s"Training ${$(algorithm).name}...")
        model.train($(corpus))
        println(s"Saving model to '${$(outputDir)}'...")
        model.save($(outputDir))
        println("Done.")
      }
    })
    .add(new Command[Unit] {
      override val name: String = "evaluate"
      override val help: String = "Evaluates an existing model on a new dataset."

      algorithm.register
      corpus.register

      val testSize: Param[Int] = Param("test-size")
        .help("Number of conversations to use for the test set.")
        .default(500)
        .register

      val resultsFile: Param[File] = Param("results-file")
        .help("File to store JSON-formatted evaluation results in.")
        .default { new File($(corpus).path.getParent + "/" + $(algorithm).name + "/results.json") }
        .register

      override def exec(): Unit = {
        println("Generating test/train split...")
        val (testDocs: Seq[Document], trainDocs: Seq[Document]) = $(corpus).roots.splitAt($(testSize))
//        val test = Corpus(testDocs.flatMap(corpus.expand(_)), corpus.words, corpus.authors)
        val train = Corpus(trainDocs.flatMap($(corpus).expand(_)), $(corpus).words, $(corpus).authors)
        val test = Corpus(testDocs.flatMap($(corpus).expand(_)), $(corpus).words, $(corpus).authors)
        println(s"Training ${$(algorithm).name}...")
//        val alg: TMAlgorithm = $(algorithm)()
//        val model: TopicModel = alg.train(train)
        val model: TopicModel = $(algorithm).exec()
        model.train(train)
        println("Done.")
        val ll1 = model.logLikelihood(train)
        val ll2 = model.logLikelihood(test)
        val trainWords = train.wordCount
        val testWords = $(corpus).wordCount - train.wordCount
        println(f" Perplexity:          ${ll1 / trainWords}%4.8f")
        println(f" Left-out likelihood: ${ll2 / testWords}%4.8f")
        import edu.utulsa.util.writeJson
        writeJson($(resultsFile), Map(
          "score" -> ll1,
          "lo-score" -> (ll2 - ll1),
          "test-size" -> $(testSize),
          "train-num-docs" -> train.size,
          "test-num-docs" -> ($(corpus).size - train.size)
        ))
      }
    })
    .add(new Command[Unit] {
      override def name = "evaluate-convs"
      override def help = "Evaluate the topic model against conversations of a particular depth."

      algorithm.register
      corpus.register

      val convDepth: Param[Int] = Param("depth")
        .help("Depth of conversations to use for training set.")
        .default(2)
        .register

      val resultsFile: Param[File] = Param("results-file")
        .help("File to store JSON-formatted evaluation results in.")
        .default { new File($(corpus).path.getParent + "/" + $(algorithm).name + "/results.json") }
        .register

      def split(corpus: Corpus, depth: Int): Corpus = {
        val docs: Seq[Document] = corpus.roots.flatMap(corpus.expand(_, depth = depth))
        Corpus(docs, corpus.words, corpus.authors)
      }
      override def exec(): Unit = {
        println(s"Evaluating on conversations of depth ${$(convDepth)}.")
        println("Loading corpus...")
        val train: Corpus = split($(corpus), $(convDepth))
        val d10: Corpus = split($(corpus), 10)
        val d4: Corpus = split($(corpus), 4)
        val d5: Corpus = split($(corpus), 5)
        val trainSize: Int = train.size
        val testSize: Int = $(corpus).size - train.size
        val d10Size: Int = $(corpus).size - d10.size

        println(s"Training ${$(algorithm).name}")
        println(s" Train size: $trainSize Test size: $testSize")
        val model: TopicModel = $(algorithm).exec()
        model.train(train)

        val ll1 = model.logLikelihood(train)
        val ll2 = model.logLikelihood($(corpus))
        val lld10 = model.logLikelihood(d10)
        val lld4 = model.logLikelihood(d4)
        val lld5 = model.logLikelihood(d5)
        import edu.utulsa.util.writeJson
        writeJson($(resultsFile), Map(
          "depth" -> $(convDepth),
          "score" -> ll1,
          "lo-score" -> (ll2 - ll1),
          "perplexity" -> ((ll2 - ll1) / testSize),
          "d10-perplexity" -> ((ll2 - lld10) / d10Size),
          "train-size" -> trainSize,
          "test-size" -> testSize
        ))
        println(s" Perplexity:          ${(ll2 - ll1) / testSize}")
        println(s" Depth 10 perplexity: ${(ll2 - lld10) / d10Size}")
        println(s" Depth 5 perplexity:  ${(lld5 - lld4) / (d5.size - d4.size)}")
      }
    })
    .register

  $.parse()
  $(action).exec()
}
