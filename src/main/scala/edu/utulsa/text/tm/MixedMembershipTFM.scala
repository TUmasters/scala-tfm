package edu.utulsa.text.tm
import java.io.File

import breeze.linalg._
import breeze.numerics._
import edu.utulsa.text.{Corpus, Document, DocumentNode}
import edu.utulsa.util.Term
import edu.utulsa.util.math.{polygamma, tetragamma}
import edu.utulsa.util.math.optimize

class MixedMembershipTFM
(
  override val numTopics: Int,
  val numWords: Int,
  val numIterations: Int
) extends TopicModel(numTopics) with MMTFMParams {

  private var optim: MMTFMOptimizer = _

  val alpha: DV = DenseVector.ones[Double](numTopics) / numTopics.toDouble
  val a: DM = DenseMatrix.ones[Double](numTopics, numTopics) / numTopics.toDouble
  val eta: DV = DenseVector.rand(numWords)

  override protected def saveModel(dir: File): Unit = {
    import edu.utulsa.util.math.csvwritevec

    csvwritevec(new File(dir + "/alpha.mat"), alpha)
    csvwrite(new File(dir + "/a.mat"), a)
    csvwritevec(new File(dir + "/eta.mat"), eta)
  }
  override def train(corpus: Corpus): Unit = {
    optim = new MMTFMOptimizer(corpus, this)
    optim.fit()
  }
  override def logLikelihood(corpus: Corpus): Double = ???
}

trait MMTFMParams {
  def numTopics: Int
  def numWords: Int
  def numIterations: Int

  def alpha: DV
  def a: DM
  def eta: DV
}

private[tm] class MMTFMOptimizer(val corpus: Corpus, val params: MMTFMParams) {
  import params._

  // Constants
  val ZERO: DV = DenseVector.zeros(numTopics)

  // Variational word parameters
  val beta: DM = DenseMatrix.rand(numTopics, numWords)
  val dgBetaW: Term[Map[Int, DV]] = Term {
    (0 until numWords).map(w => w -> digamma(beta(::, w))).toMap
  }
  val dgBetaK: Term[DV] = Term {
    DenseVector((0 until numTopics).map(k => digamma(sum(beta(k, ::)))).toArray)
  }

  val nodes: Seq[DNode] = corpus.extend(new DNode(_, _))
  val roots: Seq[DNode] = nodes.filter(_.isRoot)
  val replies: Seq[DNode] = nodes.filter(!_.isRoot)
  // i take the hit now to save on performance later
  val wnodes: Map[Int, Seq[(Double, DV)]] = nodes
    .flatMap(n => n.phi)
    .groupBy(_._1)
    .map { case (w, ns) => w -> ns.map(_._2)}

  def fit(): Unit = {
    for(i <- 1 to numIterations) {
      println(s"Iteration $i")
      eStep()
      mStep()
    }
  }

  def bound: Double = {
    def dgf(x: DV): DV = digamma(x) - digamma(sum(x))
    def dirichletEnt(x: DV, y: DV): Double = lgamma(sum(x)) - sum(lgamma(x)) + sum((x-1.0) :* dgf(y))

    // CONDITIONAL LIKELIHOOD TERMS
    val g: Map[DNode, DV] = nodes.map(n => n -> (a * n.gamma) / sum(n.gamma)).toMap
    val term1: Double = roots.map(p => dirichletEnt(alpha, p.gamma)).sum
    val term2: Double = replies.map(r => dirichletEnt(g(r.parent.get), r.gamma)).sum
    val term3: Double = (0 until numTopics).map(k => dirichletEnt(eta, beta(k, ::).t)).sum
    val term4: Double = nodes.flatMap { n =>
      val dgn = dgf(n.gamma)
      n.phi.map { case (w: Int, (c: Double, phiw: DV)) =>
        (phiw dot dgn) * c
      }
    }.sum
    val bsum: Seq[Double] = (0 until numTopics).map(k => digamma(sum(beta(k, ::))))
    val b: DM = beta.mapPairs { case ((i, _), v) =>
      digamma(v) - bsum(i)
    }
    val term5: Double = nodes.flatMap { n =>
      n.phi.map { case (w: Int, (c: Double, phiw: DV)) =>
        (phiw dot b(::, w)) * c
      }
    }.sum

    // ENTROPY TERMS
    val term6: Double = nodes.map(n => dirichletEnt(n.gamma, n.gamma)).sum
    val term7: Double = nodes.flatMap { n =>
      n.phi.map { case (_, (_, phiw)) =>
        val r = phiw dot log(phiw)
//        if(r.isNaN) println(phiw)
        r
      }
    }.sum
    val term8: Double = (0 until numTopics).map(k => dirichletEnt(beta(k, ::).t, beta(k, ::).t)).sum

    println(f"$term1%12.2f $term2%12.2f $term3%12.2f $term4%12.2f $term5%12.2f $term6%12.2f $term7%12.2f $term8%12.2f")
    term1 + term2 + term3 + term4 + term5 - term6 - term7 - term8
  }

  protected def mStep(): Unit = {
    println("m-step")
    println("alpha")
    alpha := nextAlpha
    println(s" bound: $bound")
    println("a")
    a := nextA
    println(s" bound: $bound")
    println("eta")
    eta := nextEta
    println(s" bound: $bound")
  }

  protected def eStep(): Unit = {
    println("e-step")
    var iter = 0
    var e = Double.PositiveInfinity
    while(iter < 10 && e > 1e-3) {
      println(" beta")
      // update beta
//      for (k <- 0 until numTopics) beta(k, ::) := eta.t
//      for (w <- 0 until numWords)
//        if(wnodes.contains(w))
//          beta(::, w) :+= wnodes(w).map { case (c, phidj) => c * phidj }.reduce(_ + _)
//      println(s"  bound: $bound")
//      dgBetaK.reset()
//      dgBetaW.reset()
      // update per-document variational parameters
      println(s" nodes ${nodes.length} gamma & phi")
      nodes.par.foreach(_.update())
      println(s"  bound: $bound")
      println(s"  resets: ${nodes.map(_.numResets).sum}")
      iter += 1
      e = nodes.map(_.gammaDist).sum / nodes.length
      println(s"  avg. error: $e")
    }
  }

  def nextAlpha: DV = {
    val N = roots.size.toDouble
    val t2: DV = roots.map(n => digamma(n.gamma) - digamma(sum(n.gamma))).reduce(_ + _)
    val g = (x: DV) => {
      val t1: DV = N * (digamma(sum(x)) - digamma(x))
      t1 + t2
    }
    val hq = (x: DV) => -N * trigamma(x)
    val hz = (x: DV) => N * trigamma(sum(x))
    optimize.dirichletNewton(alpha, g, hq, hz)
  }

  def nextA: DM = {
    val evidence: Seq[(DV, DV, DM)] = replies.map { n =>
      val p = n.parent.get
      val pp = p.gamma :/ sum(p.gamma)
      (digamma(n.gamma) - digamma(sum(n.gamma)), pp, pp * pp.t)
    }
    val stepF = (a: DM) => {
      val gs: Seq[(DV, DV, DV, DM)] = evidence.map { case (dgn, pp, m) =>
        val g = a * pp
        (g, dgn, pp, m)
      }.toIndexedSeq
      val grad: Seq[DV] = {
        val tmp1: Seq[(DV, DV)] = gs.map { case (g, dgn, pp, _) => (digamma(sum(g)) - digamma(g) + dgn, pp) }
        (0 until numTopics).map { k =>
          tmp1.map { case (dg, pp) =>
            pp :* dg(k)
          }.reduce(_ + _)
        }
      }
      val hess: Seq[DM] = {
        // Fast inverse hessian update
        val A: DM = gs.map { case (g,_,_,m) =>
          m * trigamma(sum(g))
        }.reduce(_ + _)
        val D: Seq[DM] = (0 until numTopics).map { k =>
          gs.map { case (g, _, _, m) => - m * trigamma(g(k)) }.reduce(_ + _)
        }
        val Dinv: Seq[DM] = D.map(inv(_))
        val Dsum: DM = Dinv.reduce(_ + _)
        val Q: DM = inv(DenseMatrix.eye[Double](numTopics) + Dsum * A)
        Dinv.map(D_j => Q * D_j) // S_j
      }
      val steps: Seq[DV] = (grad zip hess).map { case (g_k: DV, s_k: DM) => s_k * g_k }
      val stepSize = steps.map(norm(_)).sum
      steps.zipWithIndex.foreach { case (step, k) =>
        val stepSize1 = norm(stepSize)
        if(stepSize1 > 0.1)
          a(k, ::) :-= step.t / norm(step) / numTopics.toDouble / 10d
        else
          a(k, ::) :-= step.t / numTopics.toDouble / 10d
      }
      if(any(a :<= 0d)) println("hmm!")
      (a, stepSize)
    }
    optimize.runOptimizer(a, stepF)
  }

  def nextEta: DV = {
    val etaConst: Double = eta(0)
    val K: Double = numTopics.toDouble
    val M: Double = numWords.toDouble
    val dgBeta = sum(digamma(beta)) - M*sum(digamma(sum(beta, Axis._1)))
    val g = (x: Double) => K*digamma(M*x) - M*digamma(x) + dgBeta
    val h = (x: Double) => K*trigamma(M*x) - M*trigamma(x)
    val newEta = optimize.uninewton(etaConst, g, h)
    DenseVector.fill(numWords, newEta)
  }

  class DNode(override val document: Document, override val index: Int) extends DocumentNode[DNode](document, index) {
    val gamma: DV = DenseVector.rand(numTopics)
    val g: DV = (a * gamma) / sum(gamma)
    var lambda: DV = DenseVector.zeros[Double](numTopics)
    val phi: Map[Int, (Double, DV)] = document.count
      .map { case (w, c) => w -> (c.toDouble, normalize(DenseVector.rand[Double](numTopics), 1.0)) }.toMap
    val M: Int = document.count.map(_._2).sum
    var gammaDist: Double = 0
    var numResets: Int = 0

    def update(): Unit = {
//      val startTime = System.currentTimeMillis()
      numResets = 0
        val (newGamma, newG, newLambda) = nextGamma
      val oldGamma: DV = gamma.copy
      gamma := newGamma
      gammaDist = norm(gamma - oldGamma)
      g := newG
      lambda := newLambda

      val tmp = digamma(gamma) - digamma(sum(gamma))
      phi.foreach { case (w: Int, (c: Double, phi_j: DV)) =>
        phi_j := nextPhi(w, tmp)
      }
//      val endTime = System.currentTimeMillis()
//      println(s" time: ${endTime-startTime}")
    }

    def nextGamma: (DV, DV, DV) = {
      var totalDist = Double.PositiveInfinity
      val newGamma = gamma.copy
      val newG = g.copy
      val newLambda = lambda.copy
      implicit val params: (DV, DV, DV) = (newGamma, newG, newLambda)
      var numIterations = 0
      while(totalDist > 1e-6 && numIterations < 20) {
        val oldGamma = newGamma.copy
        nextGammaTerm match {
          case Some(newGammaTerm) =>
            newGamma := newGammaTerm
            totalDist = norm(oldGamma - newGamma)
          case None =>
            newGamma := DenseVector.ones[Double](numTopics) * 0.1
            totalDist = Double.PositiveInfinity
            numResets += 1
        }
        newG := nextGTerm
        newLambda := nextLambdaTerm
        numIterations += 1
      }
//      println(newGamma, numIterations)
      (newGamma, newG, newLambda)
    }

    def nextGammaTerm(implicit params: (DV, DV, DV)): Option[DV] = {
      val (gamma, g, lambda) = params
      val prior: DV = parent match {
        case Some(p) => p.g
        case None => alpha
      }
      val priorSum = sum(prior)
      val gmat: DM = tile(g, 1, numTopics).t
      val gt3: DV = (gmat - a.t) * lambda
      val phiSum: DV = phi.map { case (_, (c, r)) => r * c }.fold(ZERO)(_ + _)
      val grad = (x: DV) => {
        val t1: DV = trigamma(x) :* (prior + phiSum - x)
        val t2: Double = trigamma(sum(x)) * (priorSum + M - sum(x))
        t1 - t2 - gt3
      }
      val hq = (x: DV) => {
        (tetragamma(x) :* (prior + phiSum - x)) - trigamma(x)
      }
      val hz = (x: DV) => trigamma(sum(x)) - tetragamma(sum(x)) * (priorSum + M - sum(x))
      val newGamma = optimize.dirichletNewton(gamma, grad, hq, hz, stepRatio = 0.1)
//      if(any(newGamma.map(_.isNaN))) {
//        println(g)
//        println(lambda)
//        println(gamma)
//        println(newGamma)
//        println(grad(newGamma))
//        println("wat")
//      }
      if(norm(newGamma) > 10d * numTopics || any(newGamma :< 0d) || any(newGamma.map(_.isNaN)))
        None
      else
        Some(newGamma)
    }

    def nextLambdaTerm(implicit params: (DV, DV, DV)): DV = {
      val (gamma, g, _) = params
      val num1: DV = replies.length.toDouble * (digamma(sum(g)) - digamma(g))
      val num2: DV = replies.map(r => digamma(r.gamma) - digamma(sum(r.gamma)))
        .fold(ZERO)(_ + _)
      val den: Double = sum(gamma)
      (num1 + num2) / den
    }

    def nextGTerm(implicit params: (DV, DV, DV)): DV = {
      val (gamma, _, _) = params
      a * gamma / sum(gamma)
    }

    def nextPhi(w: Int, tmp: DV = {digamma(gamma) - digamma(sum(gamma))}): DV = {
      normalize(exp(tmp), 1.0)
    }
  }
}