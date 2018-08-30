//package edu.utulsa.conversation.eval
//import breeze.numerics.log
//import edu.utulsa.conversation.text.Corpus
//import edu.utulsa.conversation.tm.TMAlgorithm
//import edu.utulsa.util.math._
//
//import scala.util.Random
//
//class CVResults(val scores: Seq[Double]) extends EvaluatorResults {
//  override def formatted = {
//    f"""
//       | Scores: ${scores.map((s) => f"$s%8.6f").mkString(", ")}
//       | Combined Score: ${scores.sum}
//     """.stripMargin
//  }
//}
//
//class CrossValidation(val numFolds: Int) extends Evaluator {
//  require(numFolds > 0, "numFolds must be positive.")
//
//  private def split(corpus: Corpus): Seq[Corpus] = {
//    rand.shuffle(corpus.toList)
//      .zipWithIndex
//      .groupBy(_._2 % numFolds)
//      .map(_._2.map(_._1))
//      //.map(_._2.flatMap(_._1.collect))
//      .map((documents) => Corpus(documents, corpus.words, corpus.authors))
//      .toSeq
//  }
//  private def evaluate(train: Corpus, test: Corpus, algorithm: TMAlgorithm): Double = {
//    val model = algorithm.train(train)
//    model.likelihood(test)
//  }
//  override def evaluate(corpus: Corpus, algorithm: TMAlgorithm): CVResults = {
//    val corpuses: Seq[Corpus] = split(corpus)
//    val scores: Seq[Double] = (0 until numFolds).map { case (index: Int) =>
//      val train: Corpus = corpuses.patch(index, Nil, 1).reduce(_ ++ _)
//      val test: Corpus = corpuses(index)
//      println(s"Training split ${index+1} (${train.size}/${test.size})")
//      evaluate(train, test, algorithm)
//    }
//    new CVResults(scores)
//  }
//}
