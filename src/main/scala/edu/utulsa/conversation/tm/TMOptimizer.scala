package edu.utulsa.conversation.tm

import edu.utulsa.conversation.text.{Corpus}

abstract class TMOptimizer[TM <: TopicModel]
(
  val corpus: Corpus,
  val numTopics: Int
) {
  def train(): TM

  val K: Int = numTopics
}
