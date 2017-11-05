package edu.utulsa

import java.io.File

package object cli {
  case class IllegalCLIArgumentException(message: String, cause: Throwable = None.orNull)
    extends Exception(message, cause)

  abstract class ParamConverter[T] {
    def decode(value: String): T
  }

  implicit object StringConverter extends ParamConverter[String] {
    def decode(value: String): String = value
  }

  implicit object IntConverter extends ParamConverter[Int] {
    def decode(value: String): Int = value.toInt
  }

  implicit object DoubleConverter extends ParamConverter[Double] {
    def decode(value: String): Double = value.toDouble
  }

  implicit object FloatConverter extends ParamConverter[Float] {
    def decode(value: String): Float = value.toFloat
  }

  implicit object FileConverter extends ParamConverter[File] {
    def decode(value: String): File = new File(value)
  }
}
