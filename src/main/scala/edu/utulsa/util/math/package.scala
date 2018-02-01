package edu.utulsa.util

import java.io.{File, PrintWriter}

import breeze.generic.{MappingUFunc, UFunc}
import breeze.linalg._
import breeze.numerics.{exp, log}
import com.sun.javaws.exceptions.InvalidArgumentException
import org.apache.commons.math3.special.Gamma

package object math {
  def lse(x: Iterable[Double]): Double = {
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

  def lop(a: DenseVector[Double], b: DenseVector[Double]) = {
    require(a.length == b.length, "Invalid dimensions: a.length != b.length")
    val k = a.length
    tile(a, 1, k) + tile(b, 1, k).t
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

  /**
    * Used from https://stackoverflow.com/a/24869852
    * @param dist
    * @tparam A
    * @return
    */
  final def sample[A](dist: Map[A, Double]): A = {
    val p = scala.util.Random.nextDouble * dist.values.sum
    val it = dist.iterator
    var accum = 0.0
    while (it.hasNext) {
      val (item, itemProb) = it.next
      accum += itemProb
      if (accum >= p)
        return item  // return so that we don't have to search through the whole distribution
    }
    sys.error(f"this should never happen")  // needed so it will compile
  }

  /**
    * Based on scipy's implementation at https://github.com/scipy/scipy/blob/v1.0.0/scipy/special/basic.py#L843
    */
  implicit object polygamma extends UFunc with MappingUFunc {
    implicit object polygammaImpl2Double extends Impl2[Int, Double, Double] {
      def apply(n: Int, x: Double): Double = {
        if(n == 0) Gamma.digamma(x)
        else Math.pow(-1.0, n+1) * Gamma.gamma(n+1.0) * Zeta.zeta(n+1, x)
      }
    }
  }

  implicit object tetragamma extends UFunc with MappingUFunc {
    implicit object tetragammaImplDouble extends Impl[Double, Double] {
      def apply(x: Double): Double = polygamma(2, x)
    }
  }
}
