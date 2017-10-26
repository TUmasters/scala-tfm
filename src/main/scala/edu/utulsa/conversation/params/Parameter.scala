package edu.utulsa.conversation.params

import edu.utulsa.conversation.params.validators

class Parameter[T] private
(
  val name: String,
  val description: String,
  val validate: T => Boolean,
  defaultValue: => T
) {
  lazy val default: Option[T] = defaultValue match {
    case null => None
    case value: T => Some(value)
  }
  def isRequired: Boolean = default.isDefined
  def isOptional: Boolean = default.isEmpty
}

object Parameter {
  def apply[T](
                name: String,
                description: String,
                validation: T => Boolean = validators.NONE,
                default: T = null
              )(implicit params: Params): Parameter[T] = {
    val param: Parameter[T] = new Parameter[T](name, description, validation, default)
    params.add(param)
    param
  }
}