//package edu.utulsa.conversation.eval
//import edu.utulsa.conversation.text.{Corpus, Document}
//import edu.utulsa.conversation.tm.TMAlgorithm
//
//import scala.util.Random
//
//
//class TTSResults(val score: Double, val trainSize: Int, val testSize: Int) extends EvaluatorResults {
//  override def formatted: String = {
//    f"""
//       | Score: $score%4.6f
//       | Train: $trainSize/${trainSize+testSize}
//      """.stripMargin
//  }
//}
//
//class TrainTestSplit(val p: Double) extends Evaluator {
//  require(p > 0 && p < 1, "p must be a value between 0 and 1.")
//  def split(corpus: Corpus): (Corpus, Corpus) = {
//    val roots = corpus.roots
//    val trainSize: Int = math.min(math.max(math.round(roots.size * p), 1), roots.size-1).toInt
//    val testSize: Int = roots.size - trainSize
//    println((p, trainSize, testSize))
//    val trainRoots: Set[Document] = rand.shuffle(roots.toList).take(trainSize).toSet
//    val testSet: Seq[Document] = roots.filterNot(trainRoots.contains).toSeq.flatMap(_.collect)
//    val trainSet: Seq[Document] = trainRoots.flatMap(_.collect).toSeq
//    (new Corpus(trainSet, corpus.words, corpus.authors),
//      new Corpus(testSet, corpus.words, corpus.authors))
//  }
//  override def evaluate(corpus: Corpus, algorithm: TMAlgorithm): TTSResults = {
//    val (trainCorpus: Corpus, testCorpus: Corpus) = split(corpus)
//    val model = algorithm.train(trainCorpus)
//    val score: Double = model.likelihood(testCorpus)
//    new TTSResults(score, trainCorpus.size, testCorpus.size)
//  }
//}
