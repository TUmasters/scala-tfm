package edu.utulsa.text

import breeze.linalg.{DenseMatrix, DenseVector, inv, norm, sum}

package object tm {
  // saves some space and I can see how good switch from 64-bit to 32-bit precision is
  type DV = DenseVector[Double]
  type DM = DenseMatrix[Double]
}
