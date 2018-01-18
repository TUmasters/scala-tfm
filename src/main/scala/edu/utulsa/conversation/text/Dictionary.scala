package edu.utulsa.conversation.text

import java.io.{File, PrintWriter}

class Dictionary(private val w2i: Map[String, Int]) extends Iterable[Int] {
  lazy private val i2w: Map[Int, String] =
    w2i
      .map { case (w, i: Int) => (i, w) }
      .toMap

  lazy val ids: Seq[Int] = w2i.values.toSeq
  lazy val keys: Set[String] = w2i.keys.toSet

  def apply(word: String): Int = w2i(word)
  def apply(words: Seq[String]): Seq[Int] = words.map(w2i)
  def apply(index: Int): String = i2w(index)

  override def size: Int = w2i.size
  override def iterator: Iterator[Int] = ids.iterator

  def save(file: File): Unit = {
    file.createNewFile()
    Some(new PrintWriter(file)).foreach { (p) =>
      w2i.foreach { case (w, i) =>
        p.write(f"$i%-10d $w\n")
      }
      p.close()
    }
  }
}

object Dictionary {
  def apply(words: Set[String]): Dictionary = {
    new Dictionary(words.zipWithIndex.toMap)
  }
  def load(file: File): Dictionary = {
    val regex = "(\\d+)\\s+(\\S+)$".r
    val buffer = io.Source.fromFile(file)
    val w2i: Map[String, Int] = buffer.getLines.map {
      case regex(id: String, word: String) => (word, id.toInt)
    }.toMap
    new Dictionary(w2i)
  }
}
