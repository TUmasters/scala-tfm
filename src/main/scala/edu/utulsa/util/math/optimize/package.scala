package edu.utulsa.util.math

import breeze.linalg._
import breeze.numerics._

package object optimize {
  /**
    * Generic specification of an optimization algorithm.
    * @param initialValue
    * @param fStep
    * @param maxIter
    * @param stepThresh
    * @return
    */
  def runOptimizer[T]
  (
    initialValue: T,
    fStep: T => (T, Double),
    maxIter: Int = 100, stepThresh: Double = 1e-2
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

  def uninewton
  (
    initialValue: Double,
    g: Double => Double,
    h: Double => Double,
    maxIter: Int = 10, stepThresh: Double = 1e-2
  ): Double = {
    val step = (x: Double) => {
      val step = g(x) / h(x)
      var newX = x - step
      if(newX <= 0d) newX = 1e-2
      (newX, norm(step))
    }
    runOptimizer(initialValue, step, maxIter, stepThresh)
  }

  def newton
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
    runOptimizer(initialValue, step, maxIter, stepThresh)
  }

  /**
    * Newton's method for optimizing Dirichlet-like distributions.
    * @param g
    * @param hq
    * @param hz
    * @return
    */
  def dirichletNewton
  (
    initialValue: DenseVector[Double],
    g: DenseVector[Double] => DenseVector[Double],
    hq: DenseVector[Double] => DenseVector[Double],
    hz: DenseVector[Double] => Double,
    maxIter: Int = 100, stepThresh: Double = 1e-6, stepRatio: Double = 1.0,
    verbose: Int = 0
  ): DenseVector[Double] = {
    val stepF = (x: DenseVector[Double]) => {
      val grad = g(x)
      if(verbose >= 10) println(f"     |g|(x) = ${norm(grad)}%.2f")
      val q = hq(x)
      val z: Double = hz(x)
      val b = sum(grad :/ q) / ((1.0 / z) + sum(q.map(1.0/_)))
      val dstep = (grad - b) :/ q
//      val dstep2 = (diag(q) + z) \ grad
//      if(norm(dstep - dstep2) > 1e-3) println("ERROR!!!")
      val step = if(norm(dstep) > stepRatio) stepRatio * dstep / norm(dstep) else stepRatio * dstep
      val newX = x + stepRatio * step
      newX := newX.map(max(_, 1e-3))
      (newX, norm(dstep))
    }
    runOptimizer(initialValue, stepF, maxIter, stepThresh)
  }

  /**
    * Newton's method for functions with diagonal hessians.
    * @param initialValue
    * @param g
    * @param h
    */
  def diagNewton
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
    runOptimizer(initialValue, stepF, maxIter, stepThresh)
  }

  def gd
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
