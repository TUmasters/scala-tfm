package edu.utulsa.util

import scala.collection.mutable

package object iterable {
  implicit class TopK[T](x: Iterator[T]) {
    def topBy[B](n: Int)(f: T => B)(implicit ord: Ordering[B]): IndexedSeq[T] = {
      implicit val ord2: Ordering[(T, B)] = Ordering.by(t => t._2)
      val queue: mutable.PriorityQueue[(T, B)] = mutable.PriorityQueue()
      x.foreach(i => {
        queue.enqueue((i, f(i)))
        if(queue.size > n)
          queue.dequeue()
      })
      queue.dequeueAll.reverse.map(_._1)
    }
    def top(n: Int)(implicit ordering: Ordering[T]): Iterable[T] = {
      topBy(n)(a => a)
    }
  }
}
