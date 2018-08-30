package edu.utulsa.text.tm

import java.io.File

import breeze.linalg._
import breeze.numerics._
import edu.utulsa.text.{Corpus, Document, DocumentNode}

class LatentDirichletAllocation
(
  override val numTopics: Int,
  val numWords: Int,
  val usesSmoothing: Boolean = false
) extends TopicModel(numTopics) with LDAParams {

  val alpha: DenseVector[Double] = DenseVector.rand(numTopics) * 5d
  var eta: Double = 0.1d
  val theta: DenseMatrix[Double] = normalize(DenseMatrix.rand(numTopics, numWords), Axis._1, 1.0)

  private var optim: LDAOptimizer = _

  override protected def saveModel(dir: File): Unit = {}

  override def logLikelihood(corpus: Corpus): Double = ???

  override def train(corpus: Corpus): Unit = {
    optim = new LDAOptimizer(this, corpus)
    optim.fit()
  }
}

trait LDAParams {
  def numTopics: Int
  def numWords: Int
  def usesSmoothing: Boolean

  def alpha: DenseVector[Double]
  var eta: Double
  def theta: DenseMatrix[Double]
}

//trait VariationalLDAParams {
//  def beta: DenseMatrix[Double]
//}

// Implementation of LDA based on Variational Inference EM algorithm
// given in http://www.cs.columbia.edu/~blei/papers/BleiLafferty2009.pdf
class LDAOptimizer(val params: LDAParams, corpus: Corpus) {
  import params._
  import edu.utulsa.util.math._

  private var nodes: Seq[DNode] = _

  private val lambda: DenseMatrix[Double] = {
    if(usesSmoothing)
      DenseMatrix.rand[Double](numTopics, numWords)
    else
      null
  }


  def fit(): this.type = {
    nodes = corpus.extend(new DNode(_, _))

    var err = Double.PositiveInfinity
    var numIter = 0
    var b1: Double = 0
    var b2: Double = 0
    while(err >= 1e-4 && numIter < 100) {
      println(s"iteration ${numIter+1}")
      println(" e-step")
      println(alpha)
      val oldAlpha = alpha.copy
      eStep(numIter)
      b1 = bound
      println(f"  bound: $b1%.4f")
      if(b1 < b2) println(" WARNING: bound function decreased in e-step!")
      println(" m-step")
      mStep()
      err = norm(alpha - oldAlpha)
      numIter += 1
      println(f"  error $err%.5f")
      b2 = bound
      println(f"  bound: $b2%.4f")
      if(b2 < b1) println(" WARNING: bound function decreased in m-step!")
    }

    this
  }

  def bound: Double = {
    def dirichletEnt(x: DenseVector[Double], y: DenseVector[Double]): Double =
      lgamma(sum(x)) - sum(lgamma(x)) + sum((x-1d) :* (digamma(y) - digamma(sum(y))))
    val c1: Double = nodes.map(n => dirichletEnt(alpha, n.gamma)).sum
    val c2: Double = {
      if (usesSmoothing)
        (0 until numTopics)
          .map(k => digamma(numWords * eta) - numWords * digamma(eta) + (eta - 1) :* sum(digamma(lambda(k, ::)) - digamma(sum(lambda(k, ::)))))
          .sum
      else 0d
    }
    val c3: Double = nodes.map { n =>
      val gEnt = digamma(n.gamma) - digamma(sum(n.gamma))
      n.phi.map { case (w: Int, c: Int, phi_j: DenseVector[Double]) =>
        sum(c.toDouble * phi_j :* gEnt)
      }.sum
    }.sum
    val c4: Double = {
      if(usesSmoothing) {
        val dgLambda: DenseVector[Double] = DenseVector((0 until numTopics).map(k => digamma(sum(lambda(k, ::)))).toArray)
        nodes.map { n =>
          n.phi.map { case (w: Int, c: Int, phi_j: DenseVector[Double]) =>
            sum(c.toDouble * phi_j :* (digamma(lambda(::, w)) - dgLambda))
          }.sum
        }.sum
      }
      else {
        nodes.map { n =>
          n.phi.map { case(w, c, phi_j) =>
              sum(c.toDouble * phi_j :* log(theta(::, w)))
          }.sum
        }.sum
      }
    }

    val q1: Double = nodes.map { n =>
      val e = dirichletEnt(n.gamma, n.gamma)
      if(e.isNaN) 0d
      else if(e.isInfinite) {
        println(n.gamma)
        0d
      }
      else e
    }.sum
    val q2: Double = {
      if(usesSmoothing)
        (0 until numTopics).map(k => dirichletEnt(lambda(k, ::).t, lambda(k, ::).t)).sum
      else 0d
    }
    val q3: Double = nodes.map { n =>
      n.phi
        .map { case (w, c, phi_j) => sum(phi_j :* log(phi_j)) }
        .sum
    }.sum

    println(f"$c1%20.4f $c2%20.4f $c3%20.4f $c4%20.4f $q1%20.4f $q2%20.4f $q3%20.4f")
    c1 + c2 + c3 + c4 - q1 - q2 - q3
  }

  def eStep(iter: Int): Unit = {
    var err = Double.PositiveInfinity
    var numIter = 0
    var (b1, b2, b3) = (0d, 0d, 0d)
    b1 = bound
    while(err >= 1e-2 && numIter < 50) {
      println(s"  e-iter ${numIter+1}")
//      println("  beta")
      if(usesSmoothing)
        lambda := nextLambda

      if (usesSmoothing) {
        val dgLambdaSum = tile(digamma(sum(lambda, Axis._0)), 1, numTopics)
        val dgLambda = digamma(lambda) - dgLambdaSum
        nodes.foreach(_.update(dgLambda))
      }
      else nodes.foreach(_.update())

      err = nodes.map(_.gammaDist).sum / nodes.size
      b2 = bound
      println(f"   bound: $b2%.4f")
      println(f"   error: $err%.4f")
      if(b2 < b1) println("  WARNING: bound function decreased during this iteration!")
      b1 = b2
      numIter += 1
    }
  }

  def mStep(): Unit = {
//    println(" alpha")
    alpha := nextAlpha
//    println(" eta")
    if(!usesSmoothing)
      theta := nextTheta
    else
      eta = nextEta
  }

  def nextAlpha: DenseVector[Double] = {
    val n = corpus.size.toDouble
    val dgGamma = nodes
      .map(n => digamma(n.gamma) - digamma(sum(n.gamma)))
      .reduce(_ + _)
    val g = (x: DV) => n*(digamma(sum(x))-digamma(x)) + dgGamma
    val hq = (x: DV) => n*trigamma(x)
    val hz = (x: DV) => -n*trigamma(sum(x))
    optimize.dirichletNewton(alpha, g, hq, hz, stepRatio = 0.9, maxIter = 100)
  }

  def nextEta: Double = {
    val k = numTopics.toDouble
    val k2 = k*k
    val k3 = k2*k
    val dgLambda = sum(sum(digamma(lambda), Axis._1) - digamma(sum(lambda, Axis._1)))
    val g = (x: Double) => k2 * (digamma(k*x) - digamma(x)) + dgLambda
    val h = (x: Double) => k3 * trigamma(k*x) - k2 * trigamma(x)
    optimize.uninewton(eta, g, h)
  }

  def nextLambda: DenseMatrix[Double] = {
    val newLambda: DenseMatrix[Double] = DenseMatrix.zeros[Double](numTopics, numWords) + eta
    for(n <- nodes) {
      n.phi.foreach { case (w, c, phi_w) =>
        newLambda(::, w) :+= phi_w * c.toDouble
      }
    }
    newLambda
  }

  def nextTheta: DenseMatrix[Double] = {
    val newTheta: DenseMatrix[Double] = DenseMatrix.zeros[Double](numTopics, numWords)
    for(n <- nodes) {
      n.phi.foreach { case(w, c, phi_w) =>
        newTheta(::, w) :+= phi_w * c.toDouble
      }
    }
    normalize(newTheta, Axis._1, 1.0)
  }

  class DNode(override val document: Document, override val index: Int) extends DocumentNode[DNode](document, index) {
    private[tm] var gammaDist: Double = Double.PositiveInfinity
    val gamma: DenseVector[Double] = DenseVector.rand(numTopics)

    // Use a sparse matrix to save on space :)
    val phi: Seq[(Int, Int, DenseVector[Double])] = {
      document.count.map { case (word, count) =>
        (word, count, DenseVector.zeros[Double](numTopics))
      }
    }

    // Part (2) of Algorithm in Figure 5 of Blei paper
    def update(dgLambda: DenseMatrix[Double] = null): Unit = {
//      val b1 = bound
      val oldGamma = gamma.copy
      val newGamma = nextGamma
//      println(s"gamma: ${norm(dGamma(newGamma))} $newGamma")
      gamma := newGamma
      gammaDist = norm(oldGamma - gamma)
//      val b2 = bound
//      if(b2 < b1) println("    WARNING: Lower bound decreased during gamma update")

      val dgGamma = digamma(gamma) - digamma(sum(gamma))
      phi.foreach { case (word, _, row) =>
        val newPhi = nextPhi(word, dgGamma, dgLambda)
//        println(s"phi: ${norm(dPhi(word, newPhi))} $newPhi")
        row := newPhi
      }
//      val b3 = bound
//      if(b3 < b2) println("    WARNING: Lower bound decreased during phi update")
    }

    def nextGamma: DenseVector[Double] = {
      if (phi.nonEmpty)
        alpha :+ phi.map { case (w, c, r) => r * c.toDouble }.reduce(_ + _)
      else
        alpha
    }

    def nextPhi(w: Int, dgGamma: DenseVector[Double], dgLambda: DenseMatrix[Double] = null): DenseVector[Double] = {
      val t = {
        if (usesSmoothing)
          dgGamma :+ dgLambda(::, w)
        else
          dgGamma :+ log(theta(::, w))
      }
      exp(t - lse(t))
//      normalize(normalize(lambda(::, w), 1.0) :* exp(dgGamma), 1.0)
    }

//    def dGamma(value: DenseVector[Double]): DenseVector[Double] = {
//      val tmp = alpha + phi.map(t => t._3 * t._2.toDouble).reduce(_ + _) - value
//      (trigamma(value) :* tmp) - (trigamma(sum(value)) :* sum(tmp))
//    }

//    def dPhi(word: Int, value: DenseVector[Double]): DenseVector[Double] = {
//      val t = digamma(gamma) - digamma(sum(gamma)) + digamma(lambda(::, word)) - digamma(sum(lambda(::, word))) - 1d
//      val lambda = log(sum(exp(t)))
//      t - log(value) - lambda
//    }
  }
}
