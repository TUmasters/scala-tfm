package edu.utulsa.conversation.params

import scala.collection.mutable

class Params(val positional: List[String], val optional: Map[String, String]) {
  private var i: Int = 0
  private var s: mutable.Map[Parameter[_], Any] = mutable.Map()
  def apply[T](param: Parameter[T]): T = {
    if(s contains param)
      s(param).asInstanceOf[T]
    else param.default match {
      case Some(value) => value
    }
  }

  def add[T](param: Parameter[T])(implicit converter: ParamConverter[T]): Unit = {
    if(param.isRequired) {
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

object Params {
  def parse(args: Array[String]): Params = ???
}