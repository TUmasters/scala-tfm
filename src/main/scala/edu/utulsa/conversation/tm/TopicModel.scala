package edu.utulsa.conversation.tm

import breeze.linalg._
import breeze.numerics.log
import edu.utulsa.conversation.text.Corpus

abstract class TopicModel(val numTopics: Int, val numWords: Int) extends MathUtils {
  def save(dir: String): Unit

  def likelihood(corpus: Corpus)
}
