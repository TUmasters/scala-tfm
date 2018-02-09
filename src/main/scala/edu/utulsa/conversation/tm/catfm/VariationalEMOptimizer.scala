package edu.utulsa.conversation.tm.catfm

import breeze.linalg._
import breeze.numerics.{exp, log}
import java.io.File

import edu.utulsa.conversation.text.{Corpus, Dictionary, Document, DocumentNode}
import edu.utulsa.conversation.tm.CATFMParams
import edu.utulsa.util.{Term, TermContainer}
import edu.utulsa.util.math._

import scala.util.Random

sealed class VariationalEMOptimizer(val corpus: Corpus, params: CATFMParams) {
  import params._
  import edu.utulsa.util.math._

  type Vector = DenseVector[Double]
  type Matrix = DenseMatrix[Double]

  val N: Int = corpus.size
  val U: Int = corpus.authors.size
  val ZERO: Vector = DenseVector.zeros(K)

  val (trees, dnodes, cnodes) = build(corpus)

  val roots: Seq[Int] = trees.map(_.root.index)

  /** LATENT VARIABLE ESTIMATES **/
  val q: Matrix     = normalize(DenseMatrix.rand[Double](K, N), Axis._1, 1.0) // k x n
  val r: Matrix     = normalize(DenseMatrix.rand[Double](C, U), Axis._1, 1.0) // g x u
  val qr: Array[Matrix] = (1 to C).map(_ => DenseMatrix.zeros[Double](K, N)).toArray // g x k x n

  /** USEFUL INTERMEDIATES **/
  val responses: Vector = {
    val m = DenseVector.zeros[Double](N)
    dnodes.foreach{ case (node) =>
      m(node.index) = node.replies.length.toDouble
    }
    m
  } // n x 1
  // Reply matrix
  val b: CSCMatrix[Double] = {
    val builder = new CSCMatrix.Builder[Double](N, N)

    dnodes.foreach((node) =>
      node.replies.foreach((child) =>
        builder.add(node.index, child.index, 1d)
      )
    )

    builder.result()
  }   // n x n
  /** Word occurrence matrix **/
  val c: CSCMatrix[Double] = {
    val builder = new CSCMatrix.Builder[Double](N, M)

    corpus.zipWithIndex.foreach { case (document, index) =>
      document.count.foreach { case (word, count) =>
        builder.add(index, word, count.toDouble)
      }
    }

    builder.result()
  }   // n x m

  def fit(numIterations: Int, maxEIterations: Int): Unit = {
//    println(f"initial ${approxLikelihood() / corpus.wordCount}%12.4f")
    (1 to numIterations).foreach { (interval) =>
//      println(f"iteration $interval%4d")
      eStep(maxEIterations)
//      println(f"$interval%4d e-step ${approxLikelihood() / corpus.wordCount}%12.4f")
      mStep()
//      println(f"$interval%4d m-step ${approxLikelihood() / corpus.wordCount}%12.4f")
    }
  }

  def eStep(maxEIterations: Int): Unit = {
    update(maxEIterations)

    trees.par.foreach { tree =>
      tree.nodes.foreach { node =>
        q(::, node.index) := !node.z
      }
    }

    cnodes.par.foreach { node =>
      r(::, node.user) := !node.y
    }

    (0 until C).foreach { (g) =>
      qr(g) := q
      cnodes.foreach { (unode) =>
        unode.documents.foreach { (dnode) =>
          qr(g)(::, dnode.index) :*= (!unode.y) (g)
        }
      }
    }
    //    println(unodes.map(n => norm((!n.y) - phi)).sum / unodes.size)
    //    println(!unodes.head.y)
  }

  def mStep(): Unit = {
    Seq(
      () => {
        // Pi Maximization
        pi.zipWithIndex.foreach { case (pi_g, g) =>
          pi_g := normalize(roots.map((index) => qr(g)(::, index)).reduce(_ + _) + 1e-3, 1.0)
        }
      },
      () => {
        phi := normalize(cnodes.map(n => !n.y).reduce(_ + _) + 1e-3, 1.0)
      },
      () => {
        // A maximization
        a.zipWithIndex.foreach { case (a_g, g) =>
          a_g := normalize(q * b * qr(g).t + (1.0 / K), Axis._1, 1.0)
        }
      },
      () => {
        // Theta maximization
        theta := DenseMatrix.ones[Double](M, K) * 0.1
        dnodes.par.foreach { node =>
          node.document.count.foreach { case (word, count) =>
            theta(word, ::) :+= (!node.z).t * count.toDouble * (1-(!node.tau)(word))
          }
        }
        theta := normalize(theta, Axis._0, 1.0)
        //        theta := normalize((q * c) + 0.1, Axis._1, 1.0).t
      },
      () => {
        rho = dnodes.map(!_._tau.sumTau).sum / corpus.wordCount

        phi2 := DenseVector.ones[Double](M) * 0.01
        dnodes.par.foreach { node =>
          node.document.count.foreach { case (word, count) =>
            phi2(word) :+= (!node.tau)(word) * count.toDouble
          }
        }
        phi2 := normalize(phi2, 1.0)
      }
    ).par.foreach { case (step) => step() }
//    println(s"pi diff: ${pi.tail.map(pi_g => norm(pi_g - pi(0))).sum / G}")
//    println(s"a diff: ${a.tail.map(a_g => norm(a_g.toDenseVector - a(0).toDenseVector)).sum / G}")
//    //    println(pi(0))
//    //    println(pi(1))
    params.reset()
  }

  def update(numIterations: Int): Unit = {
    import scala.util.control.Breaks._
    var yerror = Double.PositiveInfinity
    var iter = 0
    while(yerror > 1e-2 && iter < numIterations) {
//      println(s"  $iter")
      trees.par.foreach { tree =>
        var treeIter = 0
        while(tree.update() > 1e-2 && treeIter < 10) { treeIter += 1 }
      }
      cnodes.par.foreach { node => node.reset(); node.y.update() }
      yerror = cnodes.map(_.dist).sum / cnodes.size
      iter += 1
    }
  }

  def build(corpus: Corpus): (Seq[DTree], Seq[DNode], Seq[GNode]) = {
    val dnodes: Seq[DNode] = corpus.extend(new DNode(_, _))
    val convs: Map[DNode, Seq[DNode]] = dnodes.groupBy(_.root)
    val unodes: Seq[GNode] = convs.zipWithIndex.map { case ((root, docs), id) =>
      new GNode(id, docs)
    }.toSeq
    unodes.foreach { node =>
      for(dnode <- node.documents)
        dnode.group = node
    }
    val trees: Seq[DTree] = dnodes
      .filter(_.isRoot)
      .map(new DTree(_))

    (trees, dnodes, unodes)
  }

  def approxLikelihood(numSamples: Int = 500): Double = {
    val samples: Seq[Double] = (1 to numSamples).par.map { i =>
      val topics: Map[DNode, Int] = dnodes.map(node => node -> node._z.draw()).toMap
      val noises: Map[DNode, Map[Int, Boolean]] = dnodes.map(node => node -> node._tau.draw()).toMap
      val groups: Map[GNode, Int] = cnodes.map(node => node -> node.draw()).toMap
      val logP: Double = dnodes.par.map(_.logP(topics, groups, noises)).sum +
        cnodes.par.map(_.logP(groups)).sum
      val logQ: Double = dnodes.par.map(_.logQ(topics, noises)).sum +
        cnodes.par.map(_.logQ(groups)).sum
      logP - logQ
    }.seq
    lse(samples) - log(samples.size)
  }

  ////////////////////////////////////////////////////////////////
  // GROUP NODE
  ////////////////////////////////////////////////////////////////
  sealed class GNode(val user: Int, val documents: Seq[DNode]) extends TermContainer {
    def dist: Double = norm(!y :- oldY)

    def draw(): Int = sample(!groupProbs)

    val oldY: Vector = DenseVector.zeros[Double](C)
    val y: Term[Vector] = Term {
      //      val oldR = (!r).copy
      //      val n: Vector = DenseVector.zeros[Double](G)
      //      n := !logPhi
      val newY: Vector = DenseVector.zeros(C)
      for(c <- 0 until C) {
        val terms: Seq[Double] = documents.map { case (node) =>
          node.parent match {
            case None =>
              sum((!node.z) :* (!logPi)(c))
            case Some(p) =>
              sum(((!p.z) * (!node.z).t) :* (!logA)(c))
          }
        }
        newY(c) = (!logPhi)(c) + sum(terms)
      }
      exp(newY - lse(newY))
    }.initialize { normalize(DenseVector.rand[Double](C), 1.0) }

    val logQa: Term[Matrix] = Term {
      (0 until C).map { c =>
        (!y)(c) * (!logA)(c)
      }.reduce(_ + _)
    }

    val logQpi: Term[Vector] = Term {
      (0 until C).map { c =>
        (!y)(c) * (!logPi)(c)
      }.reduce(_ + _)
    }

    def logP(groups: Map[GNode, Int]): Double = {
      (!logPhi) (groups(this)-1)
    }

    def logQ(groups: Map[GNode, Int]): Double = {
      val c = groups(this)-1
      log((!y)(c))
    }

    val groupProbs: Term[Map[Int, Double]] = Term {
      (!y).toArray.zipWithIndex.map(t => (t._2, t._1)).toMap
    }

    override def reset(): Unit = {
      oldY := !y
      super.reset()
    }
  }


  ////////////////////////////////////////////////////////////////
  // DOCUMENT NODE
  ////////////////////////////////////////////////////////////////
  class DNode(override val document: Document, override val index: Int)
    extends DocumentNode[DNode](document, index) {

    var group: GNode = _

    def root: DNode = parent match {
      case Some(p) => p.root
      case None => this
    }

    object _tau extends TermContainer {
      def draw(): Map[Int, Boolean] = (!tau).mapValues(Random.nextDouble() < _)
      val sumTau: Term[Double] = Term {
        document.count.map { case (word, count) =>
          (!tau)(word) * count
        }.sum
      }

      val logTau: Term[Map[Int, Double]] = Term {
        document.count.map { case (word, _) =>
          val logP1 = ((!logPhi2)(word) + !logRho) * K
          val logP2 = (0 until C).map { c =>
            parent match {
              case Some(p) => (!group.y)(c) * (log((!p.z).t * a(c) * theta(word, ::).t) + !logRhoInv * K)
              case None => (!group.y)(c) * (sum((!logTheta)(word, ::).t + (!logPi)(c)) + !logRhoInv * K)
            }
          }.sum
          word -> (logP1 - lse(Seq(logP1, logP2)))
        }
      }
    }
    val tau: Term[Map[Int, Double]] = Term { (!_tau.logTau).mapValues(exp(_)) }

    object _z extends TermContainer {
      def draw(): Int = sample(!topicProbs)

      /**
        * Computes log probabilities for observing a set of words for each latent
        * class.
        */
      val logPwz: Term[Vector] = Term {
        document.count.map { case (word, count) =>
          (!logTheta) (word, ::).t * count.toDouble * (1-(!tau)(word))
        }
          .fold(ZERO)(_ + _)
      }

      val topicProbs: Term[Map[Int, Double]] = Term {
        (!z).toArray.zipWithIndex.map(t => (t._2, t._1)).toMap
      }

      val logZ: Term[Vector] = Term {
        val t1 = !logPwz
        val t2 = parent match {
          case None => !group.logQpi
          case Some(p) => (!group.logQa).t * !p.z
        }
        val t3 = replies.map { r =>
          (!group.logQa) * !r.z
        }.fold(ZERO)(_ + _)
        val tmp = t1 + t2 + t3
        tmp - lse(tmp)
      }.initialize {
        val a = log(DenseVector.rand[Double](K))
        a - lse(a)
      }
    }
    def dist: Double = z.oldValue match {
      case Some(oldValue) => norm((!z) - oldValue)
      case None => Double.PositiveInfinity
    }
    val z: Term[Vector] = Term {
      exp(!_z.logZ)
    }.initialize { normalize(DenseVector.rand(K), 1.0) }
      .storeOld

    def logP(topics: Map[DNode, Int], groups: Map[GNode, Int], noises: Map[DNode, Map[Int, Boolean]]): Double = {
      val z = topics(this)
      val y = groups(group)
      val noise = noises(this)
      val term1: Double = parent match {
        case Some(p) =>
          val zp: Int = topics(p)
          (!logA)(y)(zp, z)
        case None =>
          (!logPi)(y)(z)
      }
      val term2: Double = noise.map { case (word, isNoise) =>
        val count = document.count(word)
        if(isNoise)
          ((!logRho) + (!logPhi)(word)) * count
        else
          ((!logRhoInv) + (!logTheta)(word, z)) * count
      }.sum
      term1 + term2
    }

    def logQ(topics: Map[DNode, Int], noises: Map[DNode, Map[Int, Boolean]]): Double = {
      val z = topics(this)
      val noise = noises(this)
      val term1 = (!_z.logZ)(z)
      val term2 = noise.map { case (word, isNoise) =>
        if(isNoise) log((!tau)(word))
        else log(1-(!tau)(word))
      }.sum
      term1 + term2
    }

    def update(): Unit = {
      _tau.reset()
      _z.reset()
      tau.update()
      z.update()
    }
  }

  sealed class DTree(val root: DNode) {
    private def expand(node: DNode): Seq[DNode] = {
      val children =
        if (node.replies != null) node.replies.flatMap((child) => expand(child))
        else Seq()
      Seq(node) ++ children
    }

    private def splitDepth(nodes: Seq[DNode]): List[Seq[DNode]] = {
      if(nodes.isEmpty) List()
      else {
        val nextLevel = nodes.flatMap(_.replies)
        List(nodes) ::: splitDepth(nextLevel)
      }
    }

    def update(): Double = {
      for(level <- levels) {
        level.par.foreach { node => node.update() }
      }
      for(level <- levels.reverse) {
        level.par.foreach { node => node.update() }
      }
      nodes.map(_.dist).sum / nodes.size
    }

    lazy val levels: List[Seq[DNode]] = splitDepth(Seq(root))
    lazy val nodes: Seq[DNode] = expand(root)
    lazy val users: Seq[GNode] = nodes.map(_.group).distinct
  }
}