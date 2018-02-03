package edu.utulsa.conversation.tm

import breeze.linalg._
import breeze.numerics.{exp, log}
import java.io.File

import edu.utulsa.conversation.text.{Corpus, Dictionary, Document, DocumentNode}
import edu.utulsa.util.{Term, TermContainer}
import edu.utulsa.util.math._

class ConversationAwareTFM
(
  override val numTopics: Int,
  val numWords: Int,
  val numUserGroups: Int,
  val numIterations: Int,
  val maxEIterations: Int
) extends TopicModel(numTopics) with CATFMParams {
  val K: Int = numTopics
  val M: Int = numWords
  val G: Int = numUserGroups

  /** MODEL PARAMETERS **/
  val pi: Array[Vector]    = (1 to G).map(_ => normalize(DenseVector.rand(K))).toArray // k x g
  val phi: Vector          = normalize(DenseVector.rand(G), 1.0) // 1 x g
  val a: Array[Matrix]     = (1 to G).map(_ => normalize(DenseMatrix.rand(K, K), Axis._1, 1.0)).toArray // g x k x k
  val theta: Matrix        = normalize(DenseMatrix.rand(M, K), Axis._0, 1.0) // m x k

  private var optim: CATFMOptimize = _

  override lazy val params: Map[String, AnyVal] = super.params ++ Map(
    "num-user-groups" -> numUserGroups,
    "num-words" -> numWords,
    "num-iterations" -> numIterations,
    "max-e-iterations" -> maxEIterations
  )
  override protected def saveModel(dir: File): Unit = {
    require(optim != null, "Model must be trained before saving to file.")

    import edu.utulsa.util.math.csvwritevec

    // save parameters
    csvwrite(new File(dir + "/theta.csv"), theta)
    csvwritevec(new File(dir + f"/phi.csv"), phi)
    for(c <- 0 until numUserGroups) {
      csvwritevec(new File(dir + f"/pi.g$c%02d.csv"), pi(c))
      csvwrite(new File(dir + f"/a.g$c%02d.csv"), a(c))
    }

    // save user info
    val userGroups: Map[String, List[TPair]] = optim.unodes.map((node) => {
      var name: String = optim.corpus.authors(node.user)
      if (name == null) name = "[deleted]"
      name -> List((!node.y).toArray.zipWithIndex.map { case (p, i) => TPair(p, i) }.maxBy(_.p))
    }).toMap
    writeJson(new File(dir + f"/user-groups.json"), userGroups)

    val dTopics: Map[String, List[TPair]] = optim.dnodes.map((node) =>
      node.document.id ->
        (!node.z).data.zipWithIndex.map { case (p, i) => TPair(p, i) }.sortBy(-_.p).toList
    ).toMap
    writeJson(new File(dir + "/document-topics.json"), dTopics)

    val wTopics: Map[String, List[TPair]] = (0 until M).map((w) =>
      optim.corpus.words(w) ->
        theta(w, ::).t.toArray.zipWithIndex.map { case (p, i) => TPair(p, i) }.sortBy(-_.p).toList
    ).toMap
    writeJson(new File(dir + "/word-topics.json"), wTopics)
  }

  override def train(corpus: Corpus): Unit = {
    optim = new CATFMOptimize(corpus, this)
    optim.fit(numIterations, maxEIterations)
  }

  override def logLikelihood(corpus: Corpus): Double = {
    val infer = new CATFMOptimize(corpus, this)
    infer.eStep(100)
    infer.approxLikelihood()
  }
}

trait CATFMParams extends TermContainer {
  val M: Int
  val K: Int
  val G: Int

  val pi: Array[Vector]
  val logPi: Term[Array[Vector]] = Term {
    pi.map(log(_))
  }
  val phi: Vector
  val logPhi: Term[Vector] = Term {
    log(phi)
  }
  val logSumPi: Term[Vector] = Term {
    val sumPi: Vector = pi.zipWithIndex
      .map { case (pi_g, g) => pi_g :* phi(g) }
      .reduce(_ + _)
    log(sumPi)
  }
  val a: Array[Matrix]
  val logA: Term[Array[Matrix]] = Term {
    a.map(log(_))
  }
  // Used as a ``normalizing constant'' for the variational inference step of each user's group
  val logSumA: Term[Matrix] = Term {
    val sumA: Matrix = a.zipWithIndex
      .map { case (a_g, g) => a_g :* phi(g) }
      .reduce(_ + _)
    log(sumA)
  }
  val theta: Matrix
  val logTheta: Term[Matrix] = Term {
    log(theta)
  }
}


sealed class CATFMOptimize(val corpus: Corpus, params: CATFMParams) {
  import params._
  import edu.utulsa.util.math._

  val N: Int = corpus.size
  val U: Int = corpus.authors.size
  val ZERO: Vector = DenseVector.zeros(K)

  val (trees, dnodes, unodes) = build(corpus)

  val roots: Seq[Int] = trees.map(_.root.index)

  /** LATENT VARIABLE ESTIMATES **/
  val q: Matrix     = normalize(DenseMatrix.rand[Double](K, N), Axis._1, 1.0) // k x n
  val r: Matrix     = normalize(DenseMatrix.rand[Double](G, U), Axis._1, 1.0) // g x u
  val qr: Array[Matrix] = (1 to G).map(_ => DenseMatrix.zeros[Double](K, N)).toArray // g x k x n

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
    println(f"initial ${approxLikelihood() / corpus.wordCount}%12.4f")
    (1 to numIterations).foreach { (interval) =>
      println(f"iteration $interval%4d")
      eStep(maxEIterations)
      println(f"$interval%4d e-step ${approxLikelihood() / corpus.wordCount}%12.4f")
      mStep()
      println(f"$interval%4d m-step ${approxLikelihood() / corpus.wordCount}%12.4f")
    }
  }

  def eStep(maxEIterations: Int): Unit = {
    update(maxEIterations)

    trees.par.foreach { tree =>
      tree.nodes.foreach { node =>
        q(::, node.index) := !node.z
      }
    }

    unodes.par.foreach { node =>
      r(::, node.user) := !node.y
    }

    (0 until G).foreach { (g) =>
      qr(g) := q
      unodes.foreach { (unode) =>
        unode.documents.foreach { (dnode) =>
          qr(g)(::, dnode.index) :*= (!unode.y) (g)
        }
      }
    }
    println(!unodes.head.y)
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
        phi := normalize(unodes.map(n => !n.y).reduce(_ + _) + 1e-3, 1.0)
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
            theta(word, ::) :+= (!node.z).t * count.toDouble
          }
        }
        theta := normalize(theta, Axis._0, 1.0)
//        theta := normalize((q * c) + 0.1, Axis._1, 1.0).t
      }
    ).par.foreach { case (step) => step() }
    println(phi)
    println(pi(0))
    println(pi(1))
    params.reset()
  }

  def update(numIterations: Int): Unit = {
    import scala.util.control.Breaks._
    var yerror = Double.PositiveInfinity
    var iter = 0
    while(yerror > 1e-2 && iter < numIterations) {
      trees.par.foreach { tree =>
        var treeIter = 0
        while(tree.update() > 1e-2 && treeIter < 10) { treeIter += 1 }
      }
      unodes.par.foreach { node => node.reset(); node.y.update() }
      yerror = unodes.map(_.dist).sum / unodes.size
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

  def fastApproxLikelihood(trees: Seq[DTree]): Double = {
    trees.map(tree => tree.nodes.map(n => lse((!n.logPw) :+ log(!n.z))).sum).sum
  }

  def approxLikelihood(numSamples: Int = 500): Double = {
    val samples: Seq[Double] = (1 to numSamples).par.map { i =>
      val topics: Map[DNode, Int] = dnodes.map(node => node -> sample(!node.topicProbs)).toMap
      val groups: Map[GNode, Int] = unodes.map(node => node -> sample(!node.groupProbs)).toMap
      val logPZ: Double = dnodes.par.map(node => (!node.logPw)(topics(node))).sum
      val logZ: Double = dnodes.par.map(_.logP(topics, groups)).sum +
        unodes.par.map(_.logP(topics, groups)).sum
      val logQ: Double = dnodes.par.map(node => log((!node.z)(topics(node)))).sum +
        unodes.par.map(node => log((!node.y)(groups(node)))).sum
      val logP: Double = logPZ + logZ
      logP - logQ
    }.seq
    lse(samples) - log(samples.size)
  }

  ////////////////////////////////////////////////////////////////
  // GROUP NODE
  ////////////////////////////////////////////////////////////////
  sealed class GNode(val user: Int, val documents: Seq[DNode]) extends TermContainer {

    def dist: Double = norm(!y :- oldY)

    val oldY: Vector = DenseVector.zeros[Double](G)
    val y: Term[Vector] = Term {
      //      val oldR = (!r).copy
//      val n: Vector = DenseVector.zeros[Double](G)
//      n := !logPhi
      val newY: Vector = DenseVector.zeros(G)
      for(c <- 0 until G) {
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
    }.initialize { normalize(DenseVector.rand[Double](G), 1.0) }

    val logQa: Term[Matrix] = Term {
      (0 until G).map { c =>
        (!y)(c) * (!logA)(c)
      }.reduce(_ + _)
    }

    val logQpi: Term[Vector] = Term {
      (0 until G).map { c =>
        (!y)(c) * (!logPi)(c)
      }.reduce(_ + _)
    }

    def logP(topics: Map[DNode, Int], groups: Map[GNode, Int]): Double = {
      if(G <= 1)
        0d
      else
        (!logPhi) (groups(this)-1)
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
    extends DocumentNode[DNode](document, index) with TermContainer {

    var group: GNode = _

    def root: DNode = parent match {
      case Some(p) => p.root
      case None => this
    }

    /**
      * Computes log probabilities for observing a set of words for each latent
      * class.
      */
    val logPw: Term[Vector] = Term {
      document.count.map { case (word, count) =>
        (!logTheta)(word, ::).t * count.toDouble
      }
        .fold(ZERO)(_ + _)
    }

    val logZ: Term[Vector] = Term {
      val t1 = !logPw
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

    def dist: Double = norm(!z - oldZ)
    val oldZ: Vector = DenseVector.zeros[Double](K)
    val z: Term[Vector] = Term {
      exp(!logZ)
    }

    val topicProbs: Term[Map[Int, Double]] = Term {
      (!z).toArray.zipWithIndex.map(t => (t._2, t._1)).toMap
    }

    def logP(topics: Map[DNode, Int], groups: Map[GNode, Int]): Double = {
      val z = topics(this)
      val y = groups(group)
      val term1: Double = parent match {
        case Some(p) =>
          val zp: Int = topics(p)
          (!logA)(y)(zp, z)
        case None =>
          (!logPi)(y)(z)
      }
      term1
    }

    override def reset(): Unit = {
      oldZ := !z
      super.reset()
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
        level.par.foreach { node => node.reset(); node.z.update() }
      }
      for(level <- levels.reverse) {
        level.par.foreach { node => node.reset(); node.z.update() }
      }
      nodes.map(_.dist).sum / nodes.size
    }

    lazy val levels: List[Seq[DNode]] = splitDepth(Seq(root))
    lazy val nodes: Seq[DNode] = expand(root)
    lazy val users: Seq[GNode] = nodes.map(_.group).distinct

    def logP(topics: Map[DNode, Int], groups: Map[GNode, Int]) = {
      nodes.map(_.logP(topics, groups)).sum
    }
  }
}
