package edu.utulsa.conversation.util

import java.io.{File, PrintWriter}

import breeze.linalg._
import breeze.numerics.{exp, log}

import scala.util.Random

object math {

  val rand = new Random()

  def lse(x: Array[Double]): Double = {
    val m = x.max
    log(x.map(xi => exp(xi - m)).sum) + m
  }

  /** Computes log(A * exp(b)). Very similar to the LogSumExp method, so I call it just that. **/
  def lse(A: DenseMatrix[Double], b: DenseVector[Double]): DenseVector[Double] = {
    val m = max(b)
    log(A * exp(b :- m)) :+ m
  }

  /** Standard LogSumExp method. **/
  def lse(b: Vector[Double]): Double = {
    val m = max(b)
    log(sum(exp(b :- m))) :+ m
  }

  /** LogSumExp applied to each dimension of a set of vectors. **/
  def lse(vecs: Array[DenseVector[Double]]): DenseVector[Double] = {
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

  def randPDM(k: Int): DenseMatrix[Double] = {
    val a = lowerTriangular(DenseMatrix.rand[Double](k, k))
    a * a.t + diag(DenseVector.rand[Double](k) :* 10.0)
  }

  def csvwritevec(file: File, x: DenseVector[Double]): Unit = {
    file.createNewFile()
    Some(new PrintWriter(file)).foreach { (p) =>
      x.foreach((xi) => p.write(f"$xi%f\n"))
      p.close()
    }
  }

  def csvreadvec(file: File): DenseVector[Double] = {
    val buffer = io.Source.fromFile(file)
    val values: Seq[Double] = buffer.getLines.map(_.toDouble).toSeq
    DenseVector(values: _*)
  }
}
