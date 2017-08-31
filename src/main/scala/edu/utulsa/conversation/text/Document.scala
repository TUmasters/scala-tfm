package edu.utulsa.conversation.text

class Document
(
  val index: Int,
  val id: String,
  val author: Int,
  val words: Seq[Int]
) {
  lazy val count: Seq[(Int, Int)] = {
    words
      .groupBy(word => word)
      .map { case (word: Int, items: Seq[Int]) => (word, items.length) }
      .toSeq
  }

  var parent: Option[Document] = None
  var replies: Seq[Document] = Seq()
}
