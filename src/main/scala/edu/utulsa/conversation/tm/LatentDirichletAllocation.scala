package edu.utulsa.conversation.tm

import java.io.File

import breeze.linalg._
import breeze.numerics._
import edu.utulsa.conversation.text.{Corpus, Dictionary, Document}

class LatentDirichletAllocation
(
  override val numTopics: Int,
  override val words: Dictionary,
  override val documentInfo: Map[String, List[TPair]],
  override val wordInfo: Map[String, List[TPair]],
  val alpha: DenseVector[Double],
  val beta: DenseMatrix[Double]
) extends TopicModel(numTopics, words, documentInfo, wordInfo) {
  override protected def saveModel(dir: File): Unit = {
    import MathUtils.csvwritevec
    csvwritevec(new File(dir + "/alpha.mat"), alpha)
    csvwrite(new File(dir + "/beta.mat"), beta)
  }

  override lazy val params: Map[String, AnyVal] = super.params

  override def likelihood(corpus: Corpus): Double = ???
}

object LatentDirichletAllocation {
  def train(corpus: Corpus, numTopics: Int, numIterations: Int, maxAlphaIterations: Int): LatentDirichletAllocation =
    new LDAOptimizer(corpus, numTopics, numIterations, maxAlphaIterations)
      .train()
}

// Implementation of LDA based on Variational Inference EM algorithm
// given in http://www.cs.columbia.edu/~blei/papers/BleiLafferty2009.pdf
class LDAOptimizer
(
  override val corpus: Corpus,
  override val numTopics: Int,
  val numIterations: Int,
  val maxAlphaIterations: Int = 15
) extends TMOptimizer[LatentDirichletAllocation](corpus, numTopics) {

  import MathUtils._

  val N = corpus.size
  val M = corpus.words.size

  // Model hyperparameters
  val alpha: DenseVector[Double] = DenseVector.rand(K) :* 3.0
  val eta: Double = 1.0

  val lambda: DenseMatrix[Double] = DenseMatrix.zeros[Double](M, K)
  val beta: DenseMatrix[Double] = DenseMatrix.zeros[Double](M, K)
  val nodes: Seq[DNode] = corpus.map((document) => new DNode(document)).toSeq

  override def train(): LatentDirichletAllocation = {
    (1 to numIterations).foreach { (i) =>
      println(s" Iteration $i")
      println("  E-Step")
      eStep()
      println("  M-Step")
      mStep()
    }
    val d: Map[String, List[TPair]] = nodes.map((n) =>
      n.document.id ->
        n.gamma.data.zipWithIndex.map {
          case (p, i) => TPair(p, i)
        }.sortBy(-_.p).toList
    ).toMap
    val w: Map[String, List[TPair]] = (1 to M).map((w) =>
      corpus.words(w) ->
        beta(w, ::).t.toArray.zipWithIndex.map {
          case (p, i) => TPair(p, i)
        }.sortBy(-_.p).toList
    ).toMap
    new LatentDirichletAllocation(numTopics, corpus.words, d, w, alpha, beta)
  }

  def eStep() {
    // println("alpha")
    // println(alpha)
    lambda := DenseMatrix.ones[Double](M, K) :* 0.1
    nodes.par.foreach { case (node) =>
      node.phi.par.foreach { case (word, count, row) =>
        lambda(word, ::) :+= row.t :* count.toDouble
      }
    }
    // println("lambda[1,:] = ")
    // println(lambda(1, ::))
    val dgLambdaS = digamma(sum(lambda(::, *))).t
    nodes.foreach { case (n) => n.variationalUpdate(lambda, dgLambdaS) }
    beta := normalize(lambda, Axis._1, 1.0)
  }

  def mStep(): Unit = {
    maximizeAlpha()
  }

  class DNode(val document: Document) {
    val gamma: DenseVector[Double] = alpha.copy
    // Use a sparse matrix to save on space :)
    val phi: Seq[(Int, Int, DenseVector[Double])] = {
      document.count.map { case (word, count) =>
        (word, count, DenseVector.zeros[Double](K))
      }
    }

    // Part (2) of Algorithm in Figure 5 of Blei paper
    def variationalUpdate(lambda: DenseMatrix[Double], dgLambdaS: DenseVector[Double]): Unit = {
      // (a)
      if (phi.size > 0)
        gamma := alpha :+ phi.map { case (w, c, r) => r * (c.toDouble) }.reduce(_ + _)
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

  def maximizeAlpha(): Unit = {
    val g2 = nodes.map((n) => digamma(n.gamma) :- digamma(sum(n.gamma))).reduce(_ + _)

    def alphaStep(): Double = {
      // Basically step-for-step Newton's method as described in
      // Appendix A.2 and A.4 of
      // http://www.cs.columbia.edu/~blei/papers/BleiNgJordan2003.pdf
      // val alphaOld = alpha.copy
      // println(alpha)
      val m: Double = M.toDouble
      val h: DenseVector[Double] = trigamma(alpha) :* (-m)
      // println("h")
      // println(h)
      val z: Double = trigamma(sum(alpha))
      // println("z")
      // println(z)
      val g1: DenseVector[Double] = (digamma(sum(alpha)) - digamma(alpha)) :* m
      // println("g1")
      // println(g1)
      // println("g2")
      // println(g2)
      val g: DenseVector[Double] = g1 :+ g2
      // println("g")
      // println(g)
      val c: Double = sum(g :/ h) / ((1.0 / z) + sum(h.map(1.0 / _)))
      // println("c")
      // println(c)
      val newGrad: DenseVector[Double] = (g :- c) :/ h
      // println("newGrad")
      // println(newGrad)
      alpha :-= newGrad
      // println("newAlpha")
      // // bad method of dealing with constraints
      // alpha := alpha.map((x) => if(x > 1e-2) x else 1e-2)
      // alpha := alpha.map((x) => if(x < 10.0) x else 10.0)
      // println(alpha)
      // println(norm(newGrad))
      norm(newGrad)
    }

    var iterations = 0
    // Should be fast convergence
    while (alphaStep() > 0.01 && iterations < maxAlphaIterations)
      iterations += 1
    println(s"alpha = $alpha")
    println(s"    Iterations: $iterations")
  }
}
