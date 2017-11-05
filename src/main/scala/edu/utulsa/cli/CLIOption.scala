package edu.utulsa.cli

import edu.utulsa.cli.validators.ValidationResult

trait CLIOption

case class Param[T]
(
  name: String,
  description: Option[String] = None,
  validate: T => ValidationResult = validators.NONE[T],
  default: Option[T] = None
) extends CLIOption {

  def description(msg: String): Param[T] = copy(description = Some(msg))
  def validation(method: T => ValidationResult): Param[T] = copy(validate = method)
  def default(value: T): Param[T] = copy(default = Some(value))
  def register(implicit $: CLIParser, converter: ParamConverter[T]): this.type = {
    $.register(this)
    this
  }

  def required: Boolean = default.isEmpty
  def optional: Boolean = default.isDefined
}

case class Command
(
  name: String,
  description: Option[String],
  actions: Map[String, Action] = Map()
) extends CLIOption {

  lazy val validate: String => ValidationResult = validators.IN(actions.keys.toSeq)
  def description(content: String): Command = copy(description = Some(content))
  def action(name: String)(action: Action): Command = copy(actions = actions + (name -> action))
}

trait Action {
  def description: Option[String] = None
}
//case class command[String]
//(
//name: String,
//description: Option[String] = None,
//method: () => Unit = ???
//) extends CLIOption {
//  def description(msg: String): command[String] = copy(description = Some(msg))
//  def exec(method: => Unit): command[String] = copy(method = () => method)
//}