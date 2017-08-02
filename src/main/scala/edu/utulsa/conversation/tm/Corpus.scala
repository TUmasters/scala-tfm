package edu.utulsa.conversation.tm

import java.io.File
import java.io.PrintWriter
import org.json4s._
import org.json4s.native.JsonMethods._
import org.json4s.native.Serialization
import org.json4s.native.Serialization.{write}

class Corpus private (data: Map[String, DocumentData]) {
  val dict: Dictionary = Dictionary(
    data.map((x) => x._2.words.toSeq).flatten.toSet
  )
  val users: Dictionary = Dictionary(
    data.map((x) => x._2.author).toSet
  )

  val documents: Seq[Document] = {
    val roots = data.map(_._2).filter(_.parent == null)
      .map((item) => (item, parse(item)))
    val tails = data.map(_._2).filter(_.parent != null)
      .groupBy(_.parent)
    roots.map((x) => expand(x, tails)).flatten.toSeq
  }
  documents.zipWithIndex.foreach { case (document: Document, index: Int) =>
    document.index = index
  }

  def apply(index: Int): Document = documents(index)

  def numWords: Int = dict.length
  def numDocuments: Int = documents.length
  def numUsers: Int = users.length

  lazy val replies: Seq[Document] = documents.filter((document) => document.parent == null)
  lazy val roots: Seq[Document] = documents.filter((document) => document.parent != null)

  private def parse(item: DocumentData): Document = {
    val document = Document(item.id, item.words.map(dict(_)))
    document.author = users(item.author)
    document
  }

  private def expand(root: (DocumentData, Document), tails: Map[String, Iterable[DocumentData]]): Seq[Document] = {
    if(tails.contains(root._1.id)) {
      val children = tails(root._1.id).map((x) => (x, parse(x))).toSeq
      children.foreach((child) => child._2.parent = root._2)
      root._2.children = children.map(_._2)
      Seq(root._2) ++ children.map((x) => expand(x, tails)).flatten
    }
    else {
      Seq(root._2)
    }
  }

  def save(dir: String): Unit = {
    new File(dir + "corpus/").mkdirs()
    dict.save(dir + "corpus/dictionary.csv")
    users.save(dir + "corpus/users.csv")
    implicit val formats = Serialization.formats(NoTypeHints)
    Some(new PrintWriter(dir + "corpus/documents_ids.json"))
      .foreach { (p) => p.write(
        write(documents.map((d) => d.id).toList)
      ); p.close() }
  }
}

object Corpus {
  def build(data: Map[String, DocumentData]): Corpus = new Corpus(data)
}
