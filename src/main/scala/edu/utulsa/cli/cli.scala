package edu.utulsa

import java.io.File

import edu.utulsa.cli.validators.ValidationResult

package object cli {
  case class IllegalCLIArgumentException(message: String, cause: Throwable = None.orNull)
    extends Exception(message, cause)

  trait ParamMap[T] {
    def mapping: Map[String, T]
    val validator: T => ValidationResult = validators.IN(mapping.values.toSeq)
    implicit object ParamMapConverter extends ParamConverter[T] {
      override def decode(value: String): T = {
        mapping(value)
      }
    }
  }

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
