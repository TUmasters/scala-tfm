package edu.utulsa.text

import breeze.linalg.{DenseMatrix, DenseVector}

package object tm {
  // reproducing here to resolve bug in compilation
  type DV = DenseVector[Double]
  type DM = DenseMatrix[Double]

  // Common corpus for testing our topic models.
  // This includes a set of two replies and some overlapping words
  lazy val sampleCorpus1: Corpus = {
    val words = Dictionary(Set("a","b","c","d","e","f"))
    val authors = Dictionary(Set("chad"))
    val documents = Seq(
      new Document("p1", null, 0, Seq(0, 1, 2)),
      new Document("r1", "p1", 0, Seq(1, 2, 3)),
      new Document("p2", null, 0, Seq(3, 4, 5)),
      new Document("r2", "p2", 0, Seq(4, 5, 5))
    )
    Corpus(documents, words, authors)
  }
}
