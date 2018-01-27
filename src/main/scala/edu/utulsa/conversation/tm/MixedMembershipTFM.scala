package edu.utulsa.conversation.tm
import java.io.File

import breeze.linalg.{Axis, DenseMatrix, DenseVector, norm, normalize, sum, tile}
import breeze.numerics.{digamma, exp, trigamma}
import edu.utulsa.conversation.text.{Corpus, Document, DocumentNode}
import edu.utulsa.util.math.{polygamma, tetragamma}

class MixedMembershipTFM
(
  override val numTopics: Int,
  val numWords: Int,
  val numIterations: Int
) extends TopicModel(numTopics) with MMTFMParams {

  val alpha: Vector = DenseVector.rand(numTopics)
  val a: Matrix = DenseMatrix.rand(numTopics, numTopics)
  val eta: Vector = DenseVector.rand(numWords)

  override protected def saveModel(dir: File): Unit = ???
  override def train(corpus: Corpus): Unit = {
  }
  override def logLikelihood(corpus: Corpus): Double = ???

}

trait MMTFMParams {
  def numTopics: Int
  def numWords: Int

  def alpha: Vector
  def a: Matrix
  def eta: Vector
}

sealed class MMTFMOptimizer(val corpus: Corpus, val params: MMTFMParams) {
  import params._

  // Constants
  val ZERO: Vector = DenseVector.zeros(numTopics)

  // Variational word parameters
  val beta: Matrix = DenseMatrix.rand(numTopics, numWords)

  val nodes: Seq[DNode] = corpus.extend(new DNode(_, _))
  val roots: Seq[DNode] = nodes.filter(!_.isRoot)
  val replies: Seq[DNode] = nodes.filter(!_.isRoot)
  // i take the hit now to save on performance later
  val wnodes: Map[Int, Seq[(Double, Vector)]] = nodes
    .flatMap(n => n.phi)
    .groupBy(_._1)
    .map { case (w, ns) => w -> ns.map(_._2)}

  def fit(): Unit = ???
  def infer(): Unit = ???

  protected def mStep(): Unit = {
  }
  protected def eStep(): Unit = {
    for(i <- 1 to 5) {
      // update beta
      for (k <- 0 until numTopics) beta(k, ::) := eta.t
      for (w <- 0 until numWords)
        if(wnodes.contains(w))
          beta(::, w) := wnodes(w).map { case (c, phidj) => c * phidj }.reduce(_ + _)

      // update per-document variational parameters
      nodes.par.foreach(_.update())
    }
  }

  protected def mAlpha(): Unit = {
    val N = replies.size.toDouble
    val t2: Vector = roots.map(s => digamma(s.gamma) - digamma(sum(s.gamma))).reduce(_ + _)
    val g = (x: Vector) => {
      val t1: Vector = N * (digamma(sum(x)) - digamma(x))
      t1 + t2
    }
    val hq = (x: Vector) => -N * trigamma(x)
    val hz = (x: Vector) => N * trigamma(sum(x))
    alpha := dirichletNewton(alpha, g, hq, hz)
  }
  protected def mA(): Unit = {
    val c: Vector = replies
      .map { n =>
        val p = n.parent.get
        p.gamma / sum(n.gamma) :* (digamma(n.gamma) - digamma(sum(n.gamma)))
      }
      .reduce(_ + _)
    val stepF = (a: Matrix) => {
      var stepSize = 0d
      val dg: Seq[(DNode, Vector, Double)] = replies.map { n =>
        val p = n.parent.get
        (n, digamma(a * p.gamma / sum(p.gamma)), digamma(sum(a, Axis._1) dot p.gamma / sum(p.gamma)))
      }
      for(k <- 0 until numTopics) {
        val g = c + dg.map { case (n, dg1, dg2) =>
          val p = n.parent.get
          p.gamma :* (dg2 - dg1(k))
        }.reduce(_ + _)
        val h = dg.map { case (n, dg1, dg2) =>
          val p = n.parent.get
          (p.gamma * p.gamma.t) * (dg2 - dg1(k))
        }.reduce(_ + _)
        val step = h \ g
        // note: doing replacement inplace to save on time (even though I'm wasting it massively everywhere else)
        a(k, ::) :-= step.t
        stepSize += norm(step)
      }
      (a, stepSize)
    }
    optimize(a, stepF)
  }
  protected def mEta(): Unit = {
    // TODO: Change this to (or experiment with) a uniform prior.
    val K = numTopics.toDouble
    val t2: Vector = (0 until numTopics).map { k =>
      val b = beta(k, ::).t
      digamma(b) - digamma(sum(b))
    }.reduce(_ + _)
    val g = (x: Vector) => {
      val t1: Vector = K * (digamma(sum(x)) - digamma(x))
      t1 + t2
    }
    val hq = (x: Vector) => -K * trigamma(x)
    val hz = (x: Vector) => K * trigamma(sum(x))
    eta := dirichletNewton(eta, g, hq, hz)
  }

  class DNode(override val document: Document, override val index: Int) extends DocumentNode[DNode](document, index) {
    val gamma: Vector = DenseVector.rand(numTopics)
    val g: Vector = (a * gamma) / sum(gamma)
    var lambda: Vector = DenseVector.zeros[Double](numTopics)
    val phi: Map[Int, (Double, Vector)] = document.count
      .map { case (w, c) => w -> (c.toDouble, normalize(DenseVector.rand[Double](numTopics), 1.0)) }.toMap
    val M: Int = document.count.map(_._2).sum

    def update(): Unit = {
      var totalDist = Double.PositiveInfinity
      while(totalDist > 1e-2) {
        totalDist = updateGamma() + updateLambda() + updateGamma()
      }
      updatePhi()
    }

    def updateGamma(): Double = {
      val prior: Vector = parent match {
        case Some(p) => p.g
        case None => alpha
      }
      val priorSum = sum(prior)
      val gmat: Matrix = tile(g, 1, numTopics).t
      val gt3: Vector = (gmat - a) * lambda
      val phiSum: Vector = phi.map { case (_, (c, r)) => r * c }.reduce(_ + _)
      val grad = (x: Vector) => {
        val t1: Vector = trigamma(x) :* (prior + phiSum - x)
        val t2: Double = trigamma(sum(x)) * (priorSum + M - sum(x))
        t1 - t2 - gt3
      }
      val hq = (x: Vector) => {
        (tetragamma(x) :* (prior + phiSum - x)) - trigamma(x)
      }
      val hz = (x: Vector) => trigamma(sum(x)) - tetragamma(sum(x)) * (priorSum + M - sum(x))

      val newGamma = dirichletNewton(gamma, grad, hq, hz)
      val dist = norm(gamma - newGamma)
      gamma := newGamma
      dist
    }
    def updateLambda(): Double = {
      val num1: Vector = replies.length.toDouble * (digamma(sum(g)) - digamma(g))
      val num2: Vector = replies.map(r => digamma(r.gamma) - digamma(sum(r.gamma)))
        .fold(ZERO)(_ + _)
      val den: Double = sum(gamma)
      val newLambda = (num1 + num2) / den
      val dist = norm(lambda - newLambda)
      lambda := newLambda
      dist
    }
    def updateG(): Double = {
      val newG = a * gamma / sum(gamma)
      val dist = norm(gamma - newG)
      g := newG
      dist
    }
    def updatePhi(): Unit = {
      val t: Vector = digamma(gamma) - digamma(sum(gamma))
      phi.foreach { case (w, (c, phij)) =>
        phij := normalize(exp(c * (t + digamma(beta(w, ::).t) - digamma(sum(beta(w, ::)))) + 1.0), 1.0)
      }
    }
  }
}