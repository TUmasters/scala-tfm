package edu.utulsa.util

import scala.collection.mutable

class Term[T] private(calc: => T) {
  var value: Option[T] = None
  def reset(): Unit = {
    value = None
  }
  def update(): T = {
    reset()
    get
  }
  def get: T = synchronized {
    if(value.isEmpty) value = Some(calc)
    value.get
  }
  def unary_! : T = get
  def initialize(initialValue: => T): this.type = {
    value = Some(initialValue)
    this
  }
}

object Term {
  def apply[T](update: => T)(implicit terms: mutable.ListBuffer[Term[_]] = null): Term[T] = { // (implicit terms: mutable.ListBuffer[Term[_]])
    val term = new Term(update)
    if(terms != null) terms += term
    term
  }
}

trait TermContainer {
  implicit protected val terms: mutable.ListBuffer[Term[_]] = mutable.ListBuffer()
  private var warnEmpty = false
  def reset(): Unit = {
    if(terms.isEmpty && !warnEmpty) {
      warnEmpty = true
      System.err.println("WARNING: Attempted to reset term container with no terms.")
    }
    terms.foreach(_.reset())
  }
}
