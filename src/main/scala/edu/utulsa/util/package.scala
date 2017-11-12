package edu.utulsa

import java.io.{File, PrintWriter}

package object util {
  def writeJson[A <: AnyRef](file: File, a: A): Unit = {
    import org.json4s._
    import org.json4s.native.Serialization
    import org.json4s.native.Serialization.writePretty

    implicit val formats = Serialization.formats(NoTypeHints)

    file.createNewFile()
    Some(new PrintWriter(file))
      .foreach { (p) => p.write(writePretty(a)); p.close() }
  }
}
