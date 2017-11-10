package edu.utulsa.cli

import scala.collection.mutable

//sealed abstract class Token
//sealed case class PositionalToken(value: String)
//sealed case class OptionalToken(key: String, value: String)

class CLIParser private (val positional: Seq[String], val optional: Map[String, String]) {
  private[cli] implicit val tree: ParamTree = new ParamTree
  private[cli] var options: Map[CLIParam[_], _] = null

  def apply[T](option: CLIParam[T]): T = {
    if(options contains option)
      options(option).asInstanceOf[T]
    else option match {
      case param: Param[T] =>
        param.default match {
          case Some(value) =>
            value
          case _ => throw new IllegalArgumentException(s"Required parameter '${param.name}' not found.")
        }
      case action: Action[_] =>
        throw new IllegalArgumentException(s"Expected value for action '${action.name}'.")
    }
  }

  def parse(): Unit = {
    options = tree.parse(positional, optional)
  }

  def help(): Unit = ???
  def usage(): Unit = ???
}

class ParamTree {
  private[cli] val options: mutable.ListBuffer[CLIParam[_]] = mutable.ListBuffer()
  private[cli] val converters: mutable.Map[CLIParam[_], ParamConverter[_]] = mutable.Map()
  private[cli] val subtrees: mutable.Map[Action[_], Map[Command[_], ParamTree]] = mutable.Map()

  private[cli] def register[T](option: CLIParam[T])(implicit converter: ParamConverter[T]): Unit = {
    options += option
    converters(option) = converter
    option match {
      case action: Action[_] =>
        subtrees(action) = action.commands.map(c => c -> c.tree).toMap
      case _ =>
    }
  }

  private[cli] def parse(positional: Seq[String], optional: Map[String, String]): Map[CLIParam[_], _] = {
    var posArgs: Seq[String] = positional
    var optArgs: Map[String, String] = optional
    options.flatMap {
      case param: Param[_] =>
        val converter = converters(param)
        var value: Any = null
        if(param.required) {
          value = converter.decode(posArgs.head)
          posArgs = posArgs.tail
        }
        else {
          if(optional contains param.name)
            value = converter.decode(optional(param.name))
          optArgs = optArgs.filter { case (key, _) => key != param.name }
        }
        if(value != null) Seq(param -> value)
        else Seq()
      case action: Action[_] =>
        val converter: ParamConverter[Command[_]] = converters(action).asInstanceOf[ParamConverter[Command[_]]]
        val command: Command[_] = converter.decode(posArgs.head)
        posArgs = posArgs.tail
        val subtree = subtrees(action)(command)
        val newOptions = subtree.parse(posArgs, optArgs)
        optArgs = optArgs.filter { case (key, _) => newOptions.keys.map(_.name).toSeq.contains(key) }
        Seq(action -> command) ++ newOptions
    }.toMap
  }
}

object CLIParser {
  def parse(args: Array[String]): CLIParser = {
    val (positional, optional) = parse(args.toList, List(), Map())
    new CLIParser(positional, optional)
  }

  private val isHelp = """(--help|-h)$""".r
  private val isOptional = """--([^=]+)$""".r
  private val isOptionalEquals = """--([^=]+)=(.+)$""".r

  private def parse( rest: List[String],
                     positional: List[String],
                     optional: Map[String, String]
                   ): (List[String], Map[String, String]) = {
    rest match {
      case isHelp() :: tail =>
        parse(tail, positional, optional)
      case isOptional(key) :: value :: tail =>
        parse(tail, positional, optional + (key -> value))
      case isOptionalEquals(key, value) :: tail =>
        parse(tail, positional, optional + (key -> value))
      case value :: tail =>
        parse(tail, positional ::: List(value), optional)
      case Nil => (positional, optional)
    }
  }
}
