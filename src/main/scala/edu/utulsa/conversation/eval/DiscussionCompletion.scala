//package edu.utulsa.conversation.eval
//import edu.utulsa.conversation.text.{Corpus, Document}
//import edu.utulsa.conversation.tm.TMAlgorithm
//import edu.utulsa.conversation.extra.math._
//
//class DCResults(val scores: Seq[Double], val levels: Seq[Int], val testSizes: Seq[Int]) extends EvaluatorResults {
//  override def formatted: String = {
//    s"""
//      | Levels: ${levels.map((l) => f"$l%6d").mkString(", ")}
//      | Scores: ${scores.map((s) => f"$s%6.4f").mkString(", ")}
//      | Test sizes: ${testSizes.map((s) => f"$s%6d").mkString(", ")}
//      | Avg. Score: ${lse(scores.toArray) - scores.length}
//    """.stripMargin
//  }
//}
//
//class DiscussionCompletion(levels: Seq[Int] = Seq(0, 1, 2, 3, 4, 5)) extends Evaluator {
//  def split(corpus: Corpus, level: Int): (Corpus, Corpus) = {
//    val train: Seq[Document] = corpus.roots.flatMap(_.collect { case (_, l) => l <= level})
//    val test: Seq[Document] = corpus.roots.flatMap(_.collect { case (_, l) => l > level})
//    (new Corpus(train, corpus.words, corpus.authors), new Corpus(test, corpus.words, corpus.authors))
//  }
//  override def evaluate(corpus: Corpus, algorithm: TMAlgorithm): DCResults = {
//    val (scores: Seq[Double], testSizes: Seq[Int]) = levels.map { (level) =>
//      val (train, rest) = split(corpus, level)
//      val model = algorithm.train(train)
//      val score1 = model.score
//      val score2 = model.likelihood(train ++ rest)
//      (score2 - score1, rest.size)
//    }.unzip
//    new DCResults(scores, levels, testSizes)
//  }
//}
