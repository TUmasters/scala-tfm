package edu.utulsa.conversation.text

class Document
(
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

  def collect: List[Document] = {
    if(replies.nonEmpty)
      List(this) ::: replies.map(_.collect).reduce(_ ::: _)
    else
      List(this)
  }

  def collect(filter: (Document, Int) => Boolean, level: Int = 0): List[Document] = {
    (if(filter(this, level)) List(this) else List()) ::: replies.map(_.collect(filter, level+1)).reduce(_ ::: _)
  }
}
