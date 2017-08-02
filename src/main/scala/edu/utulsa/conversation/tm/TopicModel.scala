package edu.utulsa.conversation.tm

import breeze.linalg._
import breeze.numerics.log

abstract class TopicModel(val corpus: Corpus) extends TopicUtils {
  def train(): this.type
  def save(dir: String): Unit

  protected var K: Int = 10
  def setK(k: Int): this.type = {
    this.K = k
    this
  }

  protected var numIterations: Int = 100
  def setNumIterations(numIterations: Int): this.type = {
    this.numIterations = numIterations
    this
  }
  // def topicDist(topic: Int): TopicDist
  // def documentDist(document: Int): DocumentDist
  // def documentDists: Seq[DocumentDist] =
  //   (0 until corpus.documents.length).map((document) => documentDist(document))
  // class DocumentDist(val document: Int, val topicDist: DenseVector[Double]) {
  //   type TopicProb = (Int, Double)
  //   lazy val best: TopicProb = {
  //     val the = topicDist.toArray.zipWithIndex
  //       .reduce((a, b) => if(a._1 > b._1) a else b)
  //     (the._2, the._1)
  //   }
  //   lazy val dist: Seq[TopicProb] = {
  //     topicDist.toArray.zipWithIndex
  //       .sortBy(-_._1)
  //       .map { case (p,i) => (i,p) }
  //   }
  // }

  // class TopicDist(val topic: Int, val wordDist: DenseVector[Double], val scaledDist: DenseVector[Double]) {
  //   type WordProb = (Double, Int)
  //   def likely(n: Int): Seq[WordProb] = {
  //     wordDist.toArray.zipWithIndex
  //       .sortBy(-_._1)
  //       .take(n)
  //   }
  //   def unique(n: Int): Array[WordProb] = {
  //     scaledDist.toArray.zipWithIndex
  //       .sortBy(-_._1)
  //       .take(n)
  //   }
  //   def scaled(n: Int): Array[WordProb] = {
  //     (scaledDist :* log(wordDist)).toArray.zipWithIndex
  //       .sortBy(-_._1)
  //       .take(n)
  //   }
  // }
}
