package edu.utulsa.conversation

package object params {
  abstract class ParamConverter[T] {
    def decode(value: String): T
  }

  implicit object StringConverter
}
