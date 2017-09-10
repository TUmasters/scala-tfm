package edu.utulsa.conversation.util

class Param[T] private (update: => T, val context: ParamCounter = null) {
  var lastUpdated = -1
  var value: T = _
  def force(): Unit = {
    lastUpdated = context.updateID
    value = update
  }
  def get(): T = synchronized {
    if(lastUpdated != context.updateID) {
      lastUpdated = context.updateID
      value = update
    }
    value
  }
  def default(initialValue: => T): this.type = {
    lastUpdated = 0
    value = initialValue
    this
  }
  def unary_! : T = get()
}

object Param {
  def apply[T](update: => T)(implicit counter: ParamCounter): Param[T] = {
    new Param(update, counter)
  }
}
