package edu.utulsa.cli

import edu.utulsa.cli.validators.ValidationResult

import scala.collection.mutable

trait CLIParam[T] {
  def name: String
  def help: Option[String]
  def validate: T => ValidationResult

  def register(implicit tree: ParamTree, converter: ParamConverter[T]): this.type = {
    tree.register(this)
    this
  }
}


case class Param[T]
(
  name: String,
  help: Option[String] = None,
  validate: T => ValidationResult = validators.NONE[T],
  defaultFunc: Option[() => T] = None
) extends CLIParam[T] {
  def help(msg: String): Param[T] = copy(help = Some(msg))
  def validation(method: T => ValidationResult): Param[T] = copy(validate = method)
  //def default(value: () => T): Param[T] = copy(defaultFunc = Some(value))
  def default(value: => T): Param[T] = copy(defaultFunc = Some(() => value))
  lazy val default: Option[T] = defaultFunc match {
    case Some(f) => Some(f())
    case None => None
  }

  def required: Boolean = defaultFunc.isEmpty
  def optional: Boolean = defaultFunc.isDefined
}


trait Command[T] {
  implicit val tree: ParamTree = new ParamTree
  def name: String
  def help: String = null
  def exec(): T
}


case class Action[T]
(
  name: String,
  help: Option[String] = None,
  commands: Seq[Command[T]] = Seq()
) extends CLIParam[Command[T]] {
  lazy val actionMap: ParamMap[Command[T]] = new ParamMap[Command[T]] {
    lazy val mapping: Map[String, Command[T]] = commands.map(a => a.name -> a).toMap
  }

  override lazy val validate: (Command[T]) => ValidationResult = actionMap.validator

  def help(content: String): Action[T] = copy(help = Some(content))
  def add(command: Command[T]): Action[T] = copy(commands = commands :+ command)

  def register(implicit tree: ParamTree): this.type = {
    tree.register(this)(actionMap.ParamMapConverter)
    this
  }
}
