package edu.utulsa.text

import java.io.File

class DocumentNode[T <: DocumentNode[T]](val document: Document, val index: Int) {
  private[text] var theParent: Option[T] = None
  def parent: Option[T] = theParent
  private[text] var theReplies: Seq[T] = _
  def replies: Seq[T] = theReplies
  lazy val siblings: Seq[T] = parent match {
    case Some(parent) =>
      parent.replies.filter((reply) => reply != this)
    case None =>
      Seq()
  }
  lazy val depth: Int = {
    if(replies.size <= 0) 1
    else 1 + replies.map(_.depth).max
  }
  lazy val size: Int = 1 + replies.map(_.size).fold(0)(_ + _)
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
  val parent: Map[Document, Document],
  val path: File = null
) extends Iterable[Document] {
  lazy val index: Map[Document, Int] = documents.zipWithIndex.toMap
  lazy val roots: Seq[Document] = documents.filter(_.parentId == null)
  lazy val wordCount: Int = documents.flatMap(_.count.map(_._2)).sum

  override def size: Int = documents.size
  override def iterator: Iterator[Document] = documents.iterator

  def ++(c2: Corpus): Corpus = {
    require(words == c2.words, "Corpuses do not share words.")
    require(authors == c2.authors, "Corpuses do not share authors.")
    Corpus(documents ++ c2.documents, words, authors, null)
  }

  def expand(document: Document, depth: Int = Int.MaxValue): Seq[Document] = {
    if(replies.contains(document) && depth > 0)
      Seq(document) ++ replies(document).flatMap(expand(_, depth-1))
    else
      Seq(document)
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
      if(parent contains d) {
        t.theParent = Some(m(parent(d)))
      }
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
//        corpusDir.mkdirs()
//        words.save(new File(corpusDir + "/words.csv"))
//        authors.save(new File(corpusDir + "/authors.csv"))
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
    Corpus(documents, words, authors, file)
  }

  def apply(documents: Seq[Document], words: Dictionary, authors: Dictionary, path: File = null): Corpus = {
    val m: Map[String, Document] = documents.map(d => d.id -> d).toMap
    val replies: Map[Document, Seq[Document]] = m.values.filter(_.parentId != null).groupBy(_.parentId).map {
      case (p, r) => m(p) -> r.toSeq
    }
    val parent: Map[Document, Document] = m.values.filter(_.parentId != null).map(d => d -> m(d.parentId)).toMap
    new Corpus(m.values.toSeq, words, authors, replies, parent, path)
  }
}
