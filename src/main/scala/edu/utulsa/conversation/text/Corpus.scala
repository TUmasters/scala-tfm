package edu.utulsa.conversation.text

import java.io.{File, PrintWriter}

import org.json4s._
import org.json4s.native.Serialization
import org.json4s.native.Serialization.write

sealed case class DocumentData(id: String, words: Seq[String], parent: String, author: String)

class Corpus
(
  val documents: Seq[Document],
  val words: Dictionary,
  val authors: Dictionary
) extends Iterable[Document] {
  lazy val roots: Seq[Document] = documents.filter(_.parent.isEmpty)
  lazy val replies: Seq[Document] = documents.filter(_.parent.isDefined)
  lazy val id2doc: Map[String, Document] = documents.map((d) => d.id -> d).toMap

  override def size: Int = documents.size
  override def iterator = documents.iterator

  def ++(c2: Corpus): Corpus = {
    require(words == c2.words, "Corpuses do not share words.")
    require(authors == c2.authors, "Corpuses do not share authors.")
    new Corpus(documents ++ c2.documents, words, authors)
  }
}

object Corpus {
  def load(file: File): Corpus = {
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
    if(corpusDir.exists) {
      val words = Dictionary.load(new File(corpusDir + "/words.csv"))
      val authors = Dictionary.load(new File(corpusDir + "/authors.csv"))
      Corpus(data, words, authors)
    }
    else {
      val tmp = Corpus.build(data)
      corpusDir.mkdir()
      tmp.words.save(new File(corpusDir + "/words.csv"))
      tmp.authors.save(new File(corpusDir + "/authors.csv"))
      tmp
    }
  }

  private def apply(data: Seq[DocumentData], words: Dictionary, authors: Dictionary): Corpus = {
    val documents: Seq[Document] = {
      val documents = data.zipWithIndex.map {
        case (data: DocumentData, index: Int) =>
          require(authors.keys.contains(data.author),
            s"Author '${data.author}' not found in cached 'authors.csv', documents file may have been updated. " +
              s"Please delete 'corpus' directory and re-execute.")
          data.words.foreach { w =>
            require(words.keys.contains(w),
              s"Word '$w' not found in cached 'words.csv', documents file may have been updated. " +
                s"Please delete 'corpus' directory and re-execute.")
          }
          new Document(index, data.id, authors(data.author), words(data.words))
      }
      val m: Map[String, Document] = documents.map((d) => d.id -> d).toMap
      documents.zip(data).foreach { case (document, data: DocumentData) =>
        if(data.parent != null)
          document.parent = Some(m(data.parent))
      }
      documents.filter(_.parent.isDefined).groupBy(_.parent).foreach {
        case (parent, subdocuments) => parent.get.replies = subdocuments
      }
      documents
    }
    new Corpus(documents, words, authors)
  }

  private def build(data: Seq[DocumentData]): Corpus = {
    val words = Dictionary(data.flatMap(_.words).toSet)
    val authors = Dictionary(data.map(_.author).toSet)
    this(data, words, authors)
  }
}
