package edu.utulsa.conversation.text

import java.io.{File, PrintWriter}

import org.json4s._
import org.json4s.native.Serialization
import org.json4s.native.Serialization.write

class DocumentNode[T <: DocumentNode[T]](val document: Document, val index: Int) {
  sealed var theParent: Option[T] = None
  def parent: Option[T] = theParent
  sealed var theReplies: Seq[T] = _
  def replies: Seq[T] = theReplies

  def isRoot: Boolean = parent match {
    case None => true
    case Some(_) => false
  }
}

class Corpus private (
  val documents: Seq[Document],
  val words: Dictionary,
  val authors: Dictionary,
  val replies: Map[Document, Seq[Document]],
  val parent: Map[Document, Document]
) extends Iterable[Document] {
//  lazy val roots: Seq[Document] = documents.filter(_.parent.isEmpty)
//  lazy val replies: Seq[Document] = documents.filter(_.parent.isDefined)
//  lazy val id2doc: Map[String, Document] = documents.map((d) => d.id -> d).toMap
  lazy val index: Map[Document, Int] = documents.zipWithIndex.toMap

  override def size: Int = documents.size
  override def iterator: Iterator[Document] = documents.iterator

  def ++(c2: Corpus): Corpus = {
    require(words == c2.words, "Corpuses do not share words.")
    require(authors == c2.authors, "Corpuses do not share authors.")
    Corpus(documents ++ c2.documents, words, authors)
  }

  /**
    * Auto-builds tree-structured conversations for any generic class wrapper around a document. Many of the inference
    * algorithms here need to be aware of tree-based methods, but it can be a bit of a mess. I move the complicated
    * parts here.
    * @param f Function that maps a document into a new node instance.
    * @tparam T The object to map the corpus into.
    * @return A sequence of DocumentNodes corresponding to each document.
    */
  def extend[T <: DocumentNode[T]](f: (Document, Int) => T): Seq[T] = {
    val n: Seq[T] = documents.zipWithIndex.map { case (d, i) => f(d, i) }
    val m: Map[Document, T] = n.map(t => t.document -> t).toMap
    n.foreach { t =>
      val d = t.document
      t.theParent = Some(m(parent(d)))
      t.theReplies =
        if(replies.contains(d)) replies(d).map(m)
        else Seq()
    }
    n
  }
}

object Corpus {
  def load(file: File): Corpus = {
    case class DocumentData(id: String, words: Seq[String], parent: String, author: String)
    import org.json4s._
    import org.json4s.native.JsonMethods._
    import org.json4s.native.Serialization
    import org.json4s.native.Serialization.write

    val dir: File = file.getParentFile
    case class JSONDocument(id: String, words: List[String], parent: String, author: String)
    implicit val formats = DefaultFormats
    val content = scala.io.Source.fromFile(file).mkString
    val json = parse(content).extract[List[JSONDocument]]
    val data: Seq[DocumentData] = json
      .map((document) => DocumentData(document.id, document.words, document.parent, document.author))

    val corpusDir = new File(dir + "/corpus/")
    val (words, authors) =
      if(corpusDir.exists) {
        val words = Dictionary.load(new File(corpusDir + "/words.csv"))
        val authors = Dictionary.load(new File(corpusDir + "/authors.csv"))
        (words, authors)
      }
      else {
        val words = Dictionary(data.flatMap(_.words).toSet)
        val authors = Dictionary(data.map(_.author).toSet)
        (words, authors)
      }
    val documents: Seq[Document] = data.map {
      case (data: DocumentData) =>
        require(authors.keys.contains(data.author),
          s"Author '${data.author}' not found in cached 'authors.csv', documents file may have been updated. " +
            s"Please delete 'corpus' directory and re-execute.")
        data.words.foreach { w =>
          require(words.keys.contains(w),
            s"Word '$w' not found in cached 'words.csv', documents file may have been updated. " +
              s"Please delete 'corpus' directory and re-execute.")
        }
        new Document(data.id, data.parent, authors(data.author), words(data.words))
    }
    Corpus(documents, words, authors)
  }

  def apply(documents: Seq[Document], words: Dictionary, authors: Dictionary): Corpus = {
    val m: Map[String, Document] = documents.map(d => d.id -> d).toMap
    val replies: Map[Document, Seq[Document]] = m.values.groupBy(_.parentId).map {
      case (p, r) => m(p) -> r.toSeq
    }
    val parent: Map[Document, Document] = m.values.map(d => d -> m(d.parentId)).toMap
    new Corpus(m.values.toSeq, words, authors, replies, parent)
  }
}
