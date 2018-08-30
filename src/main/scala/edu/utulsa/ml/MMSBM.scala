package edu.utulsa.ml

import breeze.linalg._
import breeze.math._

class MMSBM(val nrow: Int, val ncol: Int, val ngroups: Int) extends MMSBMParams {
  val alpha: DenseVector[Double] = DenseVector.zeros(ngroups)
  val b: DenseMatrix[Double] = DenseMatrix.rand(ngroups, ngroups)
  var rho: Double = 0.1
}

trait MMSBMParams {
  val nrow: Int
  val ncol: Int
  val ngroups: Int

  val alpha: DenseVector[Double]
  val b: DenseMatrix[Double]
  var rho: Double
}

sealed class MMSBMOptimizerVI(val params: MMSBMParams) {
  import params._

  val gammaTo: DenseMatrix[Double] = DenseMatrix.rand(ncol, ngroups)
  val gammaFrom: DenseMatrix[Double] = DenseMatrix.rand(nrow, ngroups)
  val phiTo: DenseMatrix[DenseVector[Double]] = DenseMatrix
    .fill(nrow, ncol) { DenseVector.rand(ngroups) }
    .map(normalize(_, 1.0))
  val phiFrom: DenseMatrix[DenseVector[Double]] = DenseMatrix
    .fill(nrow, ncol) { DenseVector.rand(ngroups) }
    .map(normalize(_, 1.0))

  def fit(y: DenseMatrix[Double]) = ???

  def mStep() = ???
  def nextAlpha: DenseVector[Double] = ???
  def nextB: DenseMatrix[Double] = ???
  def nextRho: Double = ???

  def eStep() = ???
  def nextGammaTo: DenseMatrix[Double] = ???
  def nextGammaFrom: DenseMatrix[Double] = ???
  def nextPhiTo(row: Int, col: Int): DenseVector[Double] = ???
  def nextPhiFrom(row: Int, col: Int): DenseVector[Double] = ???
}