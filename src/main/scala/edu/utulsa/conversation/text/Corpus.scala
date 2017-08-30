package edu.utulsa.conversation.text

import java.io.{File, PrintWriter}

import org.json4s._
import org.json4s.native.Serialization
import org.json4s.native.Serialization.write

case class DocumentData(id: String, words: Seq[String], parent: String, author: String)

class Corpus private (data: Seq[DocumentData], val words: Dictionary, val authors: Dictionary) extends Iterable[Document] {
  val all: Seq[Document] = {
    val documents = data.zipWithIndex.map {
      case (data: DocumentData, index: Int) => new Document(index, data.id, authors(data.author), words(data.words))
    }.toSeq
    val m: Map[String, Document] = documents.map((d) => d.id -> d).toMap
    documents.zip(data).foreach { case (document, data: DocumentData) =>
      document.parent = Some(m(data.parent))
    }
    documents.filter(_.parent.isDefined).groupBy(_.parent).foreach {
      case (parent, replies) => parent.get.replies = replies
    }
    documents
  }
  lazy val roots: Seq[Document] = all.filter(_.parent.isEmpty)
  lazy val replies: Seq[Document] = all.filter(_.parent.isDefined)
  lazy val id2doc: Map[String, Document] = all.map((d) => d.id -> d).toMap

  override def size: Int = all.size

  override def iterator = all.iterator
}

object Corpus {
  def build(data: Seq[DocumentData], words: Dictionary, authors: Dictionary): Corpus =
    new Corpus(data, words, authors)
}
