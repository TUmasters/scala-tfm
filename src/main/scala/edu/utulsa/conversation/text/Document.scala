package edu.utulsa.conversation.text

class Document
(
  val id: String,
  val parentId: String,
  val author: Int,
  val words: Seq[Int]
) {
  lazy val count: Seq[(Int, Int)] = {
    words
      .groupBy(word => word)
      .map { case (word: Int, items: Seq[Int]) => (word, items.length) }
      .toSeq
  }

  lazy val isRoot: Boolean = parentId == null
}
