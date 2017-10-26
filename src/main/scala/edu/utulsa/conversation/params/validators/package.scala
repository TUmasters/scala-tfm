package edu.utulsa.conversation.params

import java.io.File

package object validators {
  type Validation[T] = T => Boolean

  def NONE[T]: Validation[T] = (value: T) => true

  def AND[T](validations: Validation[T]*): Validation[T] =
    (value: T) => validations.map(_(value)).reduce(_ && _)

  def OR[T](validations: Validation[T]*): Validation[T] =
    (value: T) => validations.map(_(value)).reduce(_ || _)
  def IN[T](allowed: Seq[T]): Validation[T] =
    (value: T) => allowed contains value

  def INT_GEQ(bound: Int): Validation[Int] =
    (value: Int) => value >= bound

  def INT_LEQ(bound: Int): Validation[Int] =
    (value: Int) => value <= bound

  val IS_FILE: Validation[String] =
    (value: String) => new File(value).isFile

  val IS_DIRECTORY: Validation[String] =
    (value: String) => new File(value).isDirectory
}
