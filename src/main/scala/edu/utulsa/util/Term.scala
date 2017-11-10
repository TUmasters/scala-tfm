package edu.utulsa.util

import scala.collection.mutable

class Term[T] private(update: => T) {
  var value: Option[T] = None
  def reset(): Unit = {
    value = None
  }
  def forceUpdate(): T = {
    reset()
    get
  }
  def get: T = synchronized {
    if(value.isEmpty) value = Some(update)
    value.get
  }
  def unary_! : T = get
  def initialize(initialValue: => T): this.type = {
    value = Some(initialValue)
    this
  }
}

object Term {
  def apply[T](update: => T)(implicit terms: mutable.ListBuffer[Term[_]]): Term[T] = {
    val term = new Term(update)
    terms += term
    term
  }
}

trait TermContainer {
  implicit protected val terms: mutable.ListBuffer[Term[_]] = mutable.ListBuffer()
  def reset(): Unit = terms.foreach(_.reset())
}
