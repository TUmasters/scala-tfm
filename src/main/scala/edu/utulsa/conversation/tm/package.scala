package edu.utulsa.conversation

import breeze.linalg.{DenseMatrix, DenseVector, inv, norm, sum}

package object tm {
  // saves some space and I can see how good switch from 64-bit to 32-bit precision is
  type DV = DenseVector[Double]
  type DM = DenseMatrix[Double]

  /**
    * Generic specification of an optimization algorithm.
    * @param initialValue
    * @param fStep
    * @param maxIter
    * @param stepThresh
    * @return
    */
  private[tm] def optimize[T]
  (
  initialValue: T,
  fStep: T => (T, Double),
  maxIter: Int = 10, stepThresh: Double = 1e-2
  ): T = {
    var value = initialValue
    var iter = 0
    var stepSize = Double.PositiveInfinity
    while(iter < maxIter && stepSize > stepThresh) {
      val (newValue, step) = fStep(value)
      value = newValue
      stepSize = step
      iter += 1
    }
    value
  }

  private[tm] def newton
  (
  initialValue: DenseVector[Double],
  fg: DenseVector[Double] => DenseVector[Double],
  fh: DenseVector[Double] => DenseMatrix[Double],
  maxIter: Int = 10, stepThresh: Double = 1e-2
  ): DenseVector[Double] = {
    val step = (x: DenseVector[Double]) => {
      val g = fg(x)
      val h = fh(x)
      val step = h \ g
      (x - step, norm(step))
    }
    optimize(initialValue, step, maxIter, stepThresh)
  }

  /**
    * Newton's method for optimizing Dirichlet-like distributions.
    * @param g
    * @param hq
    * @param hz
    * @return
    */
  private[tm] def dirichletNewton
  (
    initialValue: DenseVector[Double],
    g: DenseVector[Double] => DenseVector[Double],
    hq: DenseVector[Double] => DenseVector[Double],
    hz: DenseVector[Double] => Double,
    maxIter: Int = 10, stepThresh: Double = 1e-2
  ): DenseVector[Double] = {
    val stepF = (x: DenseVector[Double]) => {
      val grad = g(x)
      val q = hq(x)
      val z: Double = hz(x)
      val b = sum(grad :/ q) / ((1.0 / z) + sum(q.map(1.0/_)))
      val step = (grad - b) :/ q
      (x - step, norm(step))
    }
    optimize(initialValue, stepF, maxIter, stepThresh)
  }

  /**
    * Newton's method for functions with diagonal hessians.
    * @param initialValue
    * @param g
    * @param h
    */
  private[tm] def diagNewton
  (
  initialValue: DenseVector[Double],
  g: DenseVector[Double] => DenseVector[Double],
  h: DenseVector[Double] => DenseVector[Double],
  maxIter: Int = 10, stepThresh: Double = 1e-2
  ): DenseVector[Double] = {
    val stepF = (x: DenseVector[Double]) => {
      val grad = g(x)
      val hess = h(x)
      val step = grad :/ hess
      (x - step, norm(step))
    }
    optimize(initialValue, stepF, maxIter, stepThresh)
  }

  private[tm] def gd
  (
  initialValue: DenseVector[Double],
  g: DenseVector[Double] => DenseVector[Double],
  s: Double, maxIter: Double, gThresh: Double = 1e-2
  ): DenseVector[Double] = {
    var iter: Int = 0
    val x: DenseVector[Double] = initialValue.copy
    var gNorm: Double = Double.PositiveInfinity
    while(iter < maxIter && gNorm > gThresh) {
      val step = g(x)
      gNorm = norm(g(x))
      x := x - s * step
    }
    x
  }
}
