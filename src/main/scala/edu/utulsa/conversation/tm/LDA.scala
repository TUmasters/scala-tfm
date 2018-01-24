package edu.utulsa.conversation.tm

import java.io.File

import breeze.linalg._
import breeze.numerics._
import edu.utulsa.conversation.text.{Corpus, Document}

class LDA
(
  override val numTopics: Int,
  val numWords: Int
) extends TopicModel(numTopics) with LDAParams {

  val alpha: DenseVector[Double] = DenseVector.rand(numTopics)
  val eta: DenseVector[Double] = DenseVector.rand(numWords)

  override protected def saveModel(dir: File): Unit = ???

  override def logLikelihood(corpus: Corpus): Double = ???

  override def train(corpus: Corpus): Unit = ???
}

trait LDAParams {
  def numTopics: Int
  def numWords: Int

  def alpha: DenseVector[Double]
  def eta: DenseVector[Double]
}

// Implementation of LDA based on Variational Inference EM algorithm
// given in http://www.cs.columbia.edu/~blei/papers/BleiLafferty2009.pdf
class LDAOptimizer(val params: LDAParams) {
  import params._
  import edu.utulsa.util.math._

//  def train(corpus: Corpus): LDAModel = {
//    val N: Int = corpus.size
//    val M: Int = corpus.words.size
//    val K: Int = $(numTopics)
//
//    // Model hyperparameters
//    val alpha: DenseVector[Double] = DenseVector.rand(K) :* 3.0
//    val eta: Double = 1.0
//
//    val lambda: DenseMatrix[Double] = DenseMatrix.zeros[Double](M, K)
//    val beta: DenseMatrix[Double] = DenseMatrix.zeros[Double](M, K)
//    val nodes: Seq[DNode] = corpus.map((document) => new DNode(document)).toSeq
//
//    def run(): LDAModel = {
//      (1 to $(numIterations)).foreach { (i) =>
//        println(s" Iteration $i")
//        println("  E-Step")
//        eStep()
//        println("  M-Step")
//        mStep()
//      }
//      val d: Map[String, List[TPair]] = nodes.map((n) =>
//        n.document.id ->
//          n.gamma.data.zipWithIndex.map {
//            case (p, i) => TPair(p, i)
//          }.sortBy(-_.p).toList
//      ).toMap
//      val w: Map[String, List[TPair]] = (1 to M).map((w) =>
//        corpus.words(w) ->
//          beta(w, ::).t.toArray.zipWithIndex.map {
//            case (p, i) => TPair(p, i)
//          }.sortBy(-_.p).toList
//      ).toMap
//      new LDAModel(K, corpus, d, w, alpha, beta)
//    }
//
//    def eStep() {
//      // println("alpha")
//      // println(alpha)
//      lambda := DenseMatrix.ones[Double](M, K) :* 0.1
//      nodes.par.foreach { case (node) =>
//        node.phi.par.foreach { case (word, count, row) =>
//          lambda(word, ::) :+= row.t :* count.toDouble
//        }
//      }
//      // println("lambda[1,:] = ")
//      // println(lambda(1, ::))
//      val dgLambdaS = digamma(sum(lambda(::, *))).t
//      nodes.foreach { case (n) => n.variationalUpdate(lambda, dgLambdaS) }
//      beta := normalize(lambda, Axis._1, 1.0)
//    }
//
//    def mStep(): Unit = {
//      maximizeAlpha()
//    }
//
    class DNode(val document: Document) {
      val gamma: DenseVector[Double] = alpha.copy

      // Use a sparse matrix to save on space :)
      val phi: Seq[(Int, Int, DenseVector[Double])] = {
        document.count.map { case (word, count) =>
          (word, count, DenseVector.zeros[Double](numTopics))
        }
      }

      // Part (2) of Algorithm in Figure 5 of Blei paper
      def update(lambda: DenseMatrix[Double], dgLambdaS: DenseVector[Double]): Unit = {
        // (a)
        if (phi.nonEmpty)
          gamma := alpha :+ phi.map { case (w, c, r) => r * c.toDouble }.reduce(_ + _)
        else
          gamma := alpha

        val digammaGamma = digamma(gamma)
        // (b)
        phi.foreach { case (word, count, row) =>
          row := digammaGamma :+ digamma(lambda(word, ::).t) :- dgLambdaS :+ log(count)
          row := exp(row :- lse(row))
        }
      }
    }

  //
//    def maximizeAlpha(): Unit = {
//      val g2 = nodes.map((n) => digamma(n.gamma) :- digamma(sum(n.gamma))).reduce(_ + _)
//
//      def alphaStep(): Double = {
//        // Basically step-for-step Newton's method as described in
//        // Appendix A.2 and A.4 of
//        // http://www.cs.columbia.edu/~blei/papers/BleiNgJordan2003.pdf
//        val m: Double = M.toDouble
//        val h: DenseVector[Double] = trigamma(alpha) :* (-m)
//        val z: Double = trigamma(sum(alpha))
//        val g1: DenseVector[Double] = (digamma(sum(alpha)) - digamma(alpha)) :* m
//        val g: DenseVector[Double] = g1 :+ g2
//        val c: Double = sum(g :/ h) / ((1.0 / z) + sum(h.map(1.0 / _)))
//        val newGrad: DenseVector[Double] = (g :- c) :/ h
//        alpha :-= newGrad
//        norm(newGrad)
//      }
//
//      var iterations = 0
//      // Should be fast convergence
//      while (alphaStep() > 0.01 && iterations < $(maxAlphaIterations))
//        iterations += 1
//      println(s"alpha = $alpha")
//      println(s"    Iterations: $iterations")
//    }
//
//    run()
//  }
}
