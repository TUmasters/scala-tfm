package edu.utulsa.cli

import edu.utulsa.cli.validators.{ValidateError, ValidateSuccess}
import scala.collection.mutable

class CLIParser(val positional: List[String], val optional: Map[String, String]) {
  private var i: Int = 0
  private var s: mutable.Map[Param[_], Any] = mutable.Map()
  def apply[T](param: Param[T]): T = {
    if(s contains param)
      s(param).asInstanceOf[T]
    else param.default match {
      case Some(value) => value
      case _ => throw new IllegalArgumentException("Attempt to access CLI param without default argument.")
    }
  }

  def register[T](param: Param[T])(implicit converter: ParamConverter[T]): Unit = {
    val value: T =
      if(param.required) converter.decode(positional(i))
      else converter.decode(optional(param.name))

    param.validate(value) match {
      case ValidateSuccess() =>
        s(param) = value
      case ValidateError(msg) =>
        throw IllegalCLIArgumentException(s"Invalid argument supplied for argument ${param.name}:\n$msg")
    }

    if(param.required) i += 1
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
        parse(tail, positional ::: List("help"), optional)
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
