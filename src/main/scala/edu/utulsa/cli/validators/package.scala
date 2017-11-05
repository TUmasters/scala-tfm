package edu.utulsa.cli

import java.io.File

package object validators {
  sealed abstract class ValidationResult
  case class ValidateSuccess() extends ValidationResult
  case class ValidateError(msg: String) extends ValidationResult

  type Validation[T] = T => ValidationResult

  private def check[T](condition: Boolean, msg: String): ValidationResult = {
    if(condition) ValidateSuccess()
    else ValidateError(msg)
  }

  def NONE[T]: Validation[T] = (value: T) => ValidateSuccess()

  def AND[T](validations: Validation[T]*): Validation[T] =
    (value: T) => {
      val msgs: Seq[String] = validations
        .map(_(value))
        .collect { case ValidateError(msg) => msg }
      if(msgs.size <= 0)
        ValidateSuccess()
      else
        ValidateError (
          msgs.zipWithIndex.map { case (i, msg) => s" ($i) $msg\n" }.mkString
        )
    }

  def IN[T](allowed: Seq[T]): Validation[T] =
    (value: T) => check(allowed contains value, {
      val allowedList = allowed.map(item => s"'$item'").mkString(", ")
      s"Unexpected value '$value'. Allowed values are: ($allowedList)"
    })

  def INT_GEQ(bound: Int): Validation[Int] =
    (value: Int) => check(value >= bound, s"Value must be greater than or equal to $bound.")

  def INT_LEQ(bound: Int): Validation[Int] =
    (value: Int) => check(value <= bound, s"Value must be less than or equal to $bound.")

  val IS_FILE: Validation[File] =
    (value: File) => check(value.isFile, s"Path must be a file.")

  val IS_DIRECTORY: Validation[File] =
    (value: File) => check(value.isDirectory, s"Path must be a directory.")
}
