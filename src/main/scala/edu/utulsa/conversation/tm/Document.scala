package edu.utulsa.conversation.tm

case class DocumentData(id: String, words: Seq[String], parent: String, author: String)

class Document(val id: String, val words: Seq[Int]) {
  lazy val count: Seq[(Int, Int)] = {
    words.groupBy(word => word).map{ case (word: Int, items: Seq[Int]) => (word, items.length) }.toSeq
  }

  var index: Int = -1
  var parent: Document = null
  var children: Seq[Document] = Seq()
  var author: Int = 0
}

object Document {
  def apply(id: String, words: Seq[Int]): Document = {
    new Document(id, words)
  }
}
