package edu.utulsa.conversation.tm

import breeze.linalg._
import breeze.numerics._
import edu.utulsa.conversation.text.{Corpus, Document}

// Implementation of LDA based on Variational Inference EM algorithm
// given in http://www.cs.columbia.edu/~blei/papers/BleiLafferty2009.pdf
class LDAModel(override val corpus: Corpus) extends TopicModel(corpus) {
  // Model hyperparameters
  var alpha: DenseVector[Double] = null
  var eta: Double = 1.0

  override def train(): this.type = {
    init()
    (1 to numIterations).foreach { (i) =>
      println(s" Iteration $i")
      println("  E-Step")
      optimize.eStep()
      println("  M-Step")
      optimize.mStep()
    }
    this
  }

  // Usually used to save the model to file -- not necessary right now.
  override def save(dir: String): Unit = ???

  protected def init(): Unit = {
    alpha = DenseVector.rand(K) :* 3.0
    optimize.init()
  }

  // Using an object for now, may use an optimizer class later like
  // Apache Spark does.
  protected object optimize {
    var lambda: DenseMatrix[Double] = null
    var beta: DenseMatrix[Double] = null

    class DNode(document: Document) {
      var gamma = alpha.copy
      // Use a sparse matrix to save on space :)
      var phi = Seq[(Int, Int, DenseVector[Double])]()

      // Part (2) of Algorithm in Figure 5 of Blei paper
      def variationalUpdate(lambda: DenseMatrix[Double], dgLambdaS: DenseVector[Double]): Unit = {
        // (a)
        if(phi.size > 0)
          gamma := alpha :+ phi.map { case (w, c, r) => r * (c.toDouble) }.reduce(_ + _)
        else
          gamma := alpha

        val digammaGamma = digamma(gamma)
        // (b)
        phi = document.count.map { case (word, count) =>
          // First assign (in log form)
          val row = digammaGamma :+ digamma(lambda(word, ::).t) :- dgLambdaS :+ log(count)
          // Now normalize (using safer LogSumExp method)
          row := exp(row :- lse(row))
          (word, count, row)
        }
      }
    }

    var nodes: Seq[DNode] = null

    def init(): Unit = {
      lambda = DenseMatrix.zeros[Double](corpus.numWords, K)
      beta = DenseMatrix.zeros(corpus.numWords, K)
      nodes = corpus.documents.map((document) => new DNode(document))
    }

    def eStep() {
      // println("alpha")
      // println(alpha)
      lambda := DenseMatrix.ones[Double](corpus.numWords, K) :* 0.1
      nodes.par.foreach { case (node) =>
        node.phi.par.foreach { case (word, count, row) =>
          lambda(word, ::) :+= row.t :* (count.toDouble)
        }
      }
      // println("lambda[1,:] = ")
      // println(lambda(1, ::))
      val dgLambdaS = digamma(sum(lambda(::, *))).t
      nodes.foreach { case (n) => n.variationalUpdate(lambda, dgLambdaS) }
      beta := normalize(lambda, Axis._1, 1.0)
    }

    def maximizeAlpha(): Unit = {
      val g2 = nodes.map((n) => digamma(n.gamma) :- digamma(sum(n.gamma))).reduce(_ + _)
      def alphaStep(): Double = {
        // Basically step-for-step Newton's method as described in
        // Appendix A.2 and A.4 of
        // http://www.cs.columbia.edu/~blei/papers/BleiNgJordan2003.pdf
        val M = corpus.numDocuments.toDouble
        // val alphaOld = alpha.copy
        // println(alpha)
        val h = trigamma(alpha) :* (-M)
        // println("h")
        // println(h)
        val z = trigamma(sum(alpha))
        // println("z")
        // println(z)
        val g1 = (digamma(sum(alpha)) - digamma(alpha)) :* M
        // println("g1")
        // println(g1)
        // println("g2")
        // println(g2)
        val g = g1 :+ g2
        // println("g")
        // println(g)
        val c = sum(g :/ h) / ((1.0 / z) + sum(h.map(1.0 / _)))
        // println("c")
        // println(c)
        val newGrad = (g :- c) :/ h
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
      while(alphaStep() > 0.01 && iterations < 1000)
        iterations += 1
      println(s"alpha = $alpha")
      println(s"    Iterations: $iterations")
    }

    def mStep(): Unit = {
      maximizeAlpha()
    }
  }
}

object LDA {
  def apply(corpus: Corpus) = new LDAModel(corpus)
}
