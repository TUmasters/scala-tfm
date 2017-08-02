package edu.utulsa.conversation.tm

import java.io.PrintWriter

class Dictionary(private val w2i: Map[String, Int]) {
  lazy private val i2w: Map[Int, String] =
    w2i
      .map { case (w: String, i: Int) => (i, w) }
      .toMap

  lazy val items: Seq[Int] = w2i.map(_._2).toSeq

  def apply(word: String): Int = w2i(word)
  def apply(index: Int): String = i2w(index)

  def length: Int = w2i.size

  def save(path: String) = {
    Some(new PrintWriter(path)).foreach { (p) =>
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
}
