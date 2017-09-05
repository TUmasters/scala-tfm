package edu.utulsa.conversation.util

class ParamCounter {
  var updateID: Int = 0
  def update(): Unit =
    this.updateID += 1
}
