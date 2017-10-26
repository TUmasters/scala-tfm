package edu.utulsa.conversation.extra

class Term[T] private(update: => T) {
  var value: Option[T] = None
  def reset(): Unit = {
    value = None
  }
  def get: T = synchronized {
    if(value.isEmpty) value = Some(update)
    value.get
  }
  def initialize(initialValue: => T): this.type = {
    value = Some(initialValue)
    this
  }
}

object Term {
  def apply[T](update: => T): Term[T] = {
    new Term(update)
  }
}
