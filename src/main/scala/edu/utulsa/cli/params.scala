package edu.utulsa.cli

import scala.collection.mutable

class params(val positional: List[String], val optional: Map[String, String]) {
  private var i: Int = 0
  private var s: mutable.Map[param[_], Any] = mutable.Map()
  def apply[T](param: param[T]): T = {
    if(s contains param)
      s(param).asInstanceOf[T]
    else param.default match {
      case Some(value) => value
    }
  }

  def add[T](param: param[T])(implicit converter: ParamConverter[T]): Unit = {
    if(param.required) {
      val value: T = converter.decode(positional(i))
      if(param.validate(value)) {
        s(param) = value
        i += 1
      }
    }
    else {
      val value: T = converter.decode(optional(param.name))
      if(param.validate(value))
        s(param) = value
    }
  }
}

object params {
  def parse(args: Array[String]): params = {
    val (positional, optional) = parse(args.toList, List(), Map())
    new params(positional, optional)
  }

  private val isHelp = """(--help|-h)""".r
  private val isOptional = """--.+""".r
  private def parse(rest: List[String], positional: List[String], optional: Map[String, String]): (List[String], Map[String, String]) = {
    rest match {
      case value :: tail if isHelp.pattern.matcher(value).matches() =>
        parse(tail, positional ::: List(value), optional)
      case key :: value :: tail if isOptional.pattern.matcher(key).matches() =>
        parse(tail, positional, optional + (key -> value))
      case value :: tail =>
        parse(tail, positional ::: List(value), optional)
      case Nil => (positional, optional)
    }
  }
}