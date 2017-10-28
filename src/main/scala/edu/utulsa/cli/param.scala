package edu.utulsa.cli

trait CLIOption

case class param[T]
(
  name: String,
  description: Option[String] = None,
  validate: T => Boolean = validators.NONE[T],
  default: Option[T] = None
) extends CLIOption {
  def description(msg: String): param[T] = copy(description = Some(msg))
  def validation(method: T => Boolean): param[T] = copy(validate = method)
  def default(value: T): param[T] = copy(default = Some(value))

  def register($: params)(implicit converter: ParamConverter[T]): this.type = {
    $.add(this)
    this
  }

  def required: Boolean = default.isEmpty
  def optional: Boolean = default.isDefined
}

case class command[String]
(
name: String,
description: Option[String] = None,
method: () => Unit = ???
) extends CLIOption {
  def exec(method: => Unit): command[String] = copy(method = () => method)
}