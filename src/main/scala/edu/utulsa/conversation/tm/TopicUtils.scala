package edu.utulsa.conversation.tm

import breeze.linalg._
import breeze.numerics.{log, log1p, exp, pow}
import scala.util.Random
import scala.util.control.Breaks._

trait TopicUtils {

  val rand = new Random()

  /** Computes log(A * exp(b))
    * In this case, we assume that b is represented using log-numbers because it
    * is small enough to lead to issues with floating point errors, but A can be
    * kept in its original form. The derivation is similation to the LogSumExp
    * derivations used for a set of log-numbers.
    */
  def lse(A: DenseMatrix[Double], b: DenseVector[Double]): DenseVector[Double] = {
    val m = max(b)
    log(A * exp(b :- m)) :+ m
  }

  def lse(b: Vector[Double]) = {
    val m = max(b)
    log(sum(exp(b :- m))) :+ m
  }

  def lse(vecs: Array[DenseVector[Double]]) = {
    val size = vecs(0).length
    val m = DenseVector[Double](
      (0 until size).map { case (i) =>
        max(vecs.map((vec) => vec(i)))
      }.toArray
    )
    log(vecs.map(_ :- m).map(exp(_)).reduce(_ + _)) :+ m
  }

  // /**
  //   * The real Schur Decomposition using the basic QR algorithm
  //   *
  //   * http://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter4.pdf
  //   */
  // def schur(a: DenseMatrix[Double]): (DenseMatrix[Double], DenseMatrix[Double]) = {
  //   val k = a.rows
  //   val (t, u) = (a.copy, DenseMatrix.eye[Double](k))
  //   breakable {
  //     while(true) {
  //       val oldT = t.copy
  //       val qr.QR(q, r) = qr(t)
  //       t := r * q
  //       u := u * q
  //       if (norm(oldT - t) < 1e-4) break
  //     }
  //   }
  //   (t, u)
  // }

  // /**
  //   * From https://people.kth.se/~eliasj/NLA/matrixeqs.pdf
  //   *
  //   * Implementation of the Bartels-Stewart algorithm
  //   */
  // def lyap(a: DenseMatrix[Double], w: DenseMatrix[Double]): DenseMatrix[Double] = {
  //   val n = a.rows
  //   val (q, t) = schur(a)
  //   val c = q.t * w * q
  //   var m = n
  //   (r.length until 1) foreach { (k) =>
  //     m = m - r(k)
  //   }
  // }

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
