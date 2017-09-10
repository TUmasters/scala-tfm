package edu.utulsa.conversation.eval

import edu.utulsa.conversation.text.Corpus
import edu.utulsa.conversation.tm.TMAlgorithm

import scala.util.Random

abstract class EvaluatorResults {
  def formatted: String
}

abstract class Evaluator {
  val rand = new Random(10)
  def evaluate(corpus: Corpus, algorithm: TMAlgorithm): EvaluatorResults
  def run(corpus: Corpus, algorithm: TMAlgorithm): Unit = {
    val results = evaluate(corpus, algorithm)
    println(results.formatted)
  }
}