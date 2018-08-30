package edu.utulsa.text.tm

import breeze.linalg._
import breeze.numerics._
import org.scalatest.FlatSpec
import org.scalatest.Matchers._

/**
  * Tests of the MMTFM optimizer.
  */
class MMTFMSuite extends FlatSpec {
  class MMTFMModel
  (
    override val numTopics: Int,
    override val numWords: Int
  ) extends MMTFMParams {
    val numIterations: Int = -1
    val alpha: DenseVector[Double] = DenseVector.zeros(numTopics)
    val a: DenseMatrix[Double] = DenseMatrix.zeros(numTopics, numTopics)
    val eta: DenseVector[Double] = DenseVector.zeros(numWords)
  }

  def genSampleOptim1(): MMTFMOptimizer = {
    val params = new MMTFMModel(2, 6)
    val optim = new MMTFMOptimizer(sampleCorpus1, params)
    params.alpha := DenseVector(1.2, 1.2)
    params.a := DenseMatrix(
      (1.5, 0.8),
      (0.7, 1.2)
    )
    params.eta := DenseVector.ones[Double](6) * 0.4
    optim.beta := DenseMatrix(
      (1.1, 0.9, 0.3, 0.8, 0.1, 0.2),
      (0.1, 0.3, 1.1, 1.0, 1.2, 0.8)
    )
    optim.nodes(0).gamma := DenseVector(1.5, 2.2)
    optim.nodes(1).gamma := DenseVector(1.8, 2.2)
    optim.nodes(2).gamma := DenseVector(1.8, 1.7)
    optim.nodes(3).gamma := DenseVector(2.3, 0.9)
    for(node <- optim.nodes) {
      implicit val params: (DV, DV, DV) = (node.gamma, node.g, node.lambda)
      node.g := node.nextGTerm
      node.lambda := node.nextLambdaTerm
    }
    optim.nodes(0).phi.foreach { case (w, (_, phi_j)) =>
      if(w == 0) phi_j := DenseVector(0.908479138627187, 0.09152086137281291)
      if(w == 1) phi_j := DenseVector(0.7302560403894699, 0.26974395961053016)
      if(w == 2) phi_j := DenseVector(0.1975031698039598, 0.8024968301960402)
    }
    optim.nodes(1).phi.foreach { case (w, (_, phi_j)) =>
      if(w == 1) phi_j := DenseVector(0.7646318439269981, 0.23536815607300188)
      if(w == 2) phi_j := DenseVector(0.22799774817038843, 0.7720022518296116)
      if(w == 3) phi_j := DenseVector(0.46418338108882523, 0.5358166189111748)
    }
    optim.nodes(2).phi.foreach { case (w, (_, phi_j)) =>
      if(w == 3) phi_j := DenseVector(0.5285481239804242, 0.4714518760195758)
      if(w == 4) phi_j := DenseVector(0.10457010069713403, 0.895429899302866)
      if(w == 5) phi_j := DenseVector(0.259449071108264, 0.740550928891736)
    }
    optim.nodes(3).phi.foreach { case (w, (_, phi_j)) =>
      if(w == 4) phi_j := DenseVector(0.21988527724665388, 0.780114722753346)
      if(w == 5) phi_j := DenseVector(0.4581673306772908, 0.5418326693227092)
    }
    optim
  }

  def dLgE(a: DV): DV = digamma(a) - digamma(sum(a))
  def ddLgE(a: DV): DV = trigamma(a) - trigamma(sum(a))

  def dAlpha(optim: MMTFMOptimizer): DV = {
    import optim._
    import params._
    roots.map(n =>
      digamma(sum(alpha)) - digamma(alpha) + dLgE(n.gamma)
    ).reduce(_ + _)
  }
  def dA(optim: MMTFMOptimizer): DM = {
    import optim._
    import params._
    replies.map { n =>
      val p: DNode = n.parent.get
      val pp: DV = p.gamma / sum(p.gamma)
      val g: DV = a * pp
      pp * (dLgE(n.gamma) - dLgE(g)).t
    }.reduce(_ + _)
  }
  def dEta(optim: MMTFMOptimizer): DV = {
    import optim._
    import params._
    (0 until numTopics).map { k =>
      -dLgE(eta) + dLgE(beta(k, ::).t)
    }.reduce(_ + _)
  }
  def dGamma(optim: MMTFMOptimizer, document: Int): DV = {
    import optim._
    import params._
    val ZERO = DenseVector.zeros[Double](numTopics)
    // We use the function including constraints because it tests gamma, g and lambda
    val n = nodes(document)
    val dGamma: DV = {
      val prior = n.parent match {
        case Some(p) => p.g
        case None => alpha
      }
      val phiSum = n.phi.map { case (_, (c, phi_j)) => phi_j * c }.reduce(_ + _)
      val term1: DV = trigamma(n.gamma) :* (prior + phiSum - n.gamma)
      val term2: Double = trigamma(sum(n.gamma)) * sum(prior + phiSum - n.gamma)
      val term3: DV = DenseVector((0 until numTopics).map(k => n.lambda dot (n.g - a(::, k))): _*)
      term1 - term2 - term3
    }
    val dG: DV = {
      val term1 = (-dLgE(n.g)) * n.replies.length.toDouble
      val term2 = n.replies.map(n => dLgE(n.gamma)).fold(ZERO)(_ + _)
      val term3 = n.lambda * sum(n.gamma)
      term1 + term2 - term3
    }
    val dLambda: DV = {
      n.g - a * n.gamma / sum(n.gamma)
    }
//    println(n.gamma, n.g, n.lambda)
//    println(dGamma, dG, dLambda)
    DenseVector.vertcat(dGamma, dG, dLambda)
  }
  def dPhi(phi_j: DV, optim: MMTFMOptimizer, document: Int, word: Int): DV = {
    import optim._
    import params._
    val n = nodes(document)
    val (c, _) = n.phi(word)
    val term1: DV = dLgE(n.gamma)
    val term2: DV = DenseVector((0 until numTopics).map { k =>
      digamma(beta(k, word)) - digamma(sum(beta(k, ::)))
    }: _*)
    val term3: DV = log(phi_j) + 1d
    val term4: Double = sum(exp(c*(term1 + term2) + 1d))
//    println(term4)
    c * (term1 + term2) - term3 - term4
  }

  "The bound function" should "compute the correct lower-bound likelihood for sampleCorpus1" in {
  }

  "The nextAlpha optimization" should "compute an approximate local maximum for sampleCorpus1" in {
    val optim = genSampleOptim1()
    optim.params.alpha := optim.nextAlpha
    norm(dAlpha(optim)) should be <= 1e-2
  }

  "The nextA function" should "compute an approximate local maximum for sampleCorpus1" in {
    val optim = genSampleOptim1()
    optim.params.a := optim.nextA
    norm(dA(optim).toDenseVector) should be <= 1e-2
  }

  "The nextEta function" should "compute an approximate local maximum for sampleCorpus1" in {
    val optim = genSampleOptim1()
    optim.params.eta := optim.nextEta
    norm(dEta(optim)) should be <= 1e-2
  }

  "The DNode.nextGamma function" should "compute an approximate local maximum for sampleCorpus1" in {
    val optim = genSampleOptim1()
    optim.nodes.zipWithIndex.foreach { case (node, id) =>
      val (oldGamma, oldG, oldLambda) = (node.gamma.copy, node.g.copy, node.lambda.copy)
      val (newGamma, newG, newLambda) = node.nextGamma
      node.gamma := newGamma
      node.g := newG
      node.lambda := newLambda
      norm(dGamma(optim, id)) should be <= 1e-4
      node.gamma := oldGamma
      node.g := oldG
      node.lambda := oldLambda
    }
  }

  "The DNode.nextPhi function" should "compute an exact local maximum for sampleCorpus1" in {
    val optim = genSampleOptim1()
    optim.nodes.zipWithIndex.foreach { case (node, id) =>
      node.phi.foreach { case (word, (_, _)) =>
        val newPhi_j = node.nextPhi(word)
        norm(dPhi(newPhi_j, optim, id, word)) should be <= 1e-4
      }
    }
  }
}
