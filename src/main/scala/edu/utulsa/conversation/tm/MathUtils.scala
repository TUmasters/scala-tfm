package edu.utulsa.conversation.tm

import breeze.linalg._
import breeze.numerics.{log, log1p, exp, pow}
import scala.util.Random
import scala.util.control.Breaks._

trait MathUtils {

  val rand = new Random()

  /** Computes log(A * exp(b)). Very similar to the LogSumExp method, so I call it just that. **/
  def lse(A: DenseMatrix[Double], b: DenseVector[Double]): DenseVector[Double] = {
    val m = max(b)
    log(A * exp(b :- m)) :+ m
  }

  /** Standard LogSumExp method. **/
  def lse(b: Vector[Double]) = {
    val m = max(b)
    log(sum(exp(b :- m))) :+ m
  }

  /** LogSumExp applied to each dimension of a set of vectors. **/
  def lse(vecs: Array[DenseVector[Double]]) = {
    // TODO: Simplify this crappy code!
    val size = vecs(0).length
    val m = DenseVector[Double](
      (0 until size).map { case (i) =>
        max(vecs.map((vec) => vec(i)))
      }.toArray
    )
    log(vecs.map(_ :- m).map(exp(_)).reduce(_ + _)) :+ m
  }

  /** Several debug message methods, to make it easier to read output. **/

  def disp(x: DenseVector[Double]): Unit = {
    print("[ ")
    x.toArray.foreach((value) => print(f"$value%4.3f "))
    println("]")
  }

  def disp(A: DenseMatrix[Double]): Unit = {
    println("[")
    A(*, ::).foreach { case (row: DenseVector[Double]) =>
      print(" ")
      disp(row)
    }
    println("]")
  }
}
