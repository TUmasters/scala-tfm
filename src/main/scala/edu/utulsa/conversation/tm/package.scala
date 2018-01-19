package edu.utulsa.conversation

import breeze.linalg.{DenseVector, norm, sum}

package object tm {
  /**
    * Solves functions in the form
    * f'(x) = g(x) + c
    * f"(x) = diag(fq(x)) + 1 1^T hz(x)
    * * @param c
    * @param g
    * @param hq
    * @param hz
    * @return
    */
  private[tm] def newton(initialValue: DenseVector[Double],
                         g: DenseVector[Double] => DenseVector[Double],
                         hq: DenseVector[Double] => DenseVector[Double],
                         hz: DenseVector[Double] => Double,
                         maxIter: Int = 10, stepThresh: Double = 1e-2): DenseVector[Double] = {
    def step(x: DenseVector[Double]): (DenseVector[Double], Double) = {
      val grad = g(x)
      val q = hq(x)
      val z: Double = hz(x)
      val b = sum(grad :/ q) / ((1.0 / z) + sum(q.map(1.0/_)))
      val step = (grad - b) :/ q
      (step, norm(step))
    }
    var value = initialValue.copy
    var iter = 0
    var stepSize = Double.PositiveInfinity
    while(iter < maxIter || stepSize > stepThresh) {
      val (newValue, newStepSize) = step(value)
      value = newValue
      stepSize = newStepSize
      iter += 1
    }
    value
  }
}
