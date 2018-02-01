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
      override val name: String = "catfm"
      override val help: String = "Conversation-Aware Topic Flow Model"

      numTopics.register
      val numUserGroups: Param[Int] = Param("num-conversation-groups")
        .help("Number of user groups for the CATFM.")
        .default(3)
        .validation(validators.INT_GEQ(1))
        .register
      numIterations.register
      val numEIterations: Param[Int] = Param("num-e-iterations")
        .help("Number of iterations to run the variational inference step of the CATFM.")
        .default(20)
        .validation(validators.INT_GEQ(1))
        .register

      override def exec(): TopicModel = {
        new ConversationAwareTFM($(numTopics), $(corpus).words.size, $(numUserGroups), $(numIterations), $(numEIterations))
      }
    })
    .add(new Command[TopicModel] {
      override val name: String = "utm"
      override val help: String = "Mixture of Unigrams (non-Bayesian)"

      numTopics.register
      numIterations.register

      override def exec(): TopicModel = {
        new UnigramTM($(numTopics), $(corpus).words.size, $(numIterations))
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
        println(s"Training ${$(algorithm).name}...")
        val startTime = System.currentTimeMillis()
        val model = $(algorithm).exec()
        model.train($(corpus))
        val endTime = System.currentTimeMillis()
        println(f"Completed in ${(endTime - startTime) / 1000.0}%.4fs")
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
//        import edu.utulsa.util.writeJson
//        writeJson($(resultsFile), Map(
//          "score" -> ll1,
//          "lo-score" -> (ll2 - ll1),
//          "test-size" -> $(testSize),
//          "train-num-docs" -> train.size,
//          "test-num-docs" -> ($(corpus).size - train.size)
//        ))
      }
    })
    .add(new Command[Unit] {
      override def name = "evaluate-depth"
      override def help = "Evaluate the topic model against conversations of a particular depth."

      algorithm.register
      corpus.register

      val startDepth: Param[Int] = Param("start-depth")
        .help("Depth of conversations to use for training set.")
        .default(2)
        .register

      val endDepth: Param[Int] = Param("end-depth")
        .help("Depth to stop testing at.")
        .default(10)
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
        println(s"Evaluating on conversations of depth ${$(startDepth)}-${$(endDepth)}.")
        println(s"Training ${$(algorithm).name} on ${$(corpus).roots.size} conversations")

        val depthInfo = ($(startDepth) to $(endDepth)).map { case depth =>
          val train: Corpus = split($(corpus), depth)
          val trainSize: Int = train.size
          val testSize: Int = $(corpus).size - train.size

          println(f"Depth: $depth%2d Train size: $trainSize%6d Test size: $testSize%6d")
          val trainWords = train.documents.flatMap(_.count.map(_._2)).sum
          val testWords = $(corpus).documents.flatMap(_.count.map(_._2)).sum - trainWords

          val model: TopicModel = $(algorithm).exec()
          model.train(train)

          val ll1 = model.logLikelihood(train)
          val ll2 = model.logLikelihood($(corpus))
          println(f" Train perplexity:    ${ll1 / trainWords}%6.4f")
          println(f" Left-out perplexity: ${(ll2 - ll1) / testWords}%6.4f")
          Map(
            "depth" -> depth,
            "train-words" -> trainWords,
            "train-size" -> trainSize,
            "test-words" -> testWords,
            "test-size" -> testSize,
            "train-perplexity" -> (ll1 / trainWords),
            "test-perplexity" -> ((ll2 - ll1) / testWords)
          )
        }

        import edu.utulsa.util.writeJson
        writeJson($(resultsFile), depthInfo)
      }
    })
    .add(new Command[Unit] {
      override def name = "evaluate-topics"
      override def help = "Evaluate the topic model with various topic configurations."

      corpus.register

      val startTopics: Param[Int] = Param("start-topics")
        .help("Depth of conversations to use for training set.")
        .default(8)
        .register

      val endTopics: Param[Int] = Param("end-topics")
        .help("Depth to stop testing at.")
        .default(50)
        .register

      val testSize: Param[Int] = Param("test-size")
        .help("Number of conversations to use in the test set.")
        .default(1000)
        .register

      def split(corpus: Corpus, depth: Int): Corpus = {
        val docs: Seq[Document] = corpus.roots.flatMap(corpus.expand(_, depth = depth))
        Corpus(docs, corpus.words, corpus.authors)
      }
      override def exec(): Unit = {
        println(s"Evaluating on topics in ${$(startTopics)} - ${$(endTopics)}.")
        println(s"Training CATFM on ${$(corpus).roots.size} conversations")

        val (testDocs: Seq[Document], trainDocs: Seq[Document]) = $(corpus).roots.splitAt($(testSize))
        val train = Corpus(trainDocs.flatMap($(corpus).expand(_)), $(corpus).words, $(corpus).authors)
        val test = Corpus(testDocs.flatMap($(corpus).expand(_)), $(corpus).words, $(corpus).authors)
        val trainWords = train.wordCount
        val testWords = test.wordCount

        println(f"Train size: ${train.size}%6d Test size: ${test.size}%6d")
        val topicInfo = Range($(startTopics), $(endTopics), 2).map { topics =>

          println(f"Topics: $topics%3d")

//          val model: TopicModel = new ConversationAwareTFM(topics, $(corpus).words.size, 3, 30, 20)
          val model: TopicModel = new ConversationAwareTFM(topics, $(corpus).words.size, 3, 30, 20)
//          val model: TopicModel = new UnigramTM(topics, $(corpus).words.size, 10)
          model.train(train)

          val ll1 = model.logLikelihood(train)
          val ll2 = model.logLikelihood($(corpus))
          println(f" Train perplexity:    ${ll1 / trainWords}%6.4f")
          println(f" Left-out perplexity: ${(ll2 - ll1) / testWords}%6.4f")
          Map(
            "num-topics" -> topics,
            "train-perplexity" -> (ll1 / trainWords),
            "test-perplexity" -> ((ll2 - ll1) / testWords)
          )
        }

//        import edu.utulsa.util.writeJson
//        writeJson($(resultsFile), depthInfo)
      }
    })
    .register

  $.parse()
  $(action).exec()
}
