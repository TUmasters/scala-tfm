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
      name -> List((!node.r).toArray.zipWithIndex.map { case (p, i) => TPair(p, i) }.maxBy(_.p))
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
    (1 to numIterations).foreach { (interval) =>
//      println(f"iteration $interval%4d")
      eStep(maxEIterations)
//      println(f"$interval%4d e-step ${approxLikelihood()}%12.4f")
      mStep()
//      println(f"$interval%4d m-step ${approxLikelihood()}%12.4f")
    }
  }

  def eStep(maxEIterations: Int): Unit = {
    update(trees, dnodes, unodes, G, maxEIterations)

    trees.par.foreach { tree =>
      tree.nodes.foreach { node =>
        q(::, node.index) := !node.z
      }
    }

    unodes.par.foreach { node =>
      r(::, node.user) := !node.r
    }

    (0 until G).foreach { (g) =>
      qr(g) := q
      unodes.foreach { (unode) =>
        unode.documents.foreach { (dnode) =>
          qr(g)(::, dnode.index) :*= (!unode.r) (g)
        }
      }
    }

    println("infer")
    println(r(::, 0))
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
        phi := normalize(unodes.map(n => !n.r).reduce(_ + _) + 1e-3, 1.0)
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
    reset()
  }

  def build(corpus: Corpus): (Seq[DTree], Seq[DNode], Seq[UNode]) = {
    val dnodes: Seq[DNode] = corpus.extend(new DNode(_, _))
    val convs: Map[DNode, Seq[DNode]] = dnodes.groupBy(_.root)
    val unodes: Seq[UNode] = convs.zipWithIndex.map { case ((root, docs), id) =>
      new UNode(id, docs)
    }.toSeq
    unodes.foreach { node =>
      for(dnode <- node.documents)
        dnode.author = node
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
      val groups: Map[UNode, Int] = unodes.map(node => node -> sample(!node.groupProbs)).toMap
      val logPZ: Double = dnodes.par.map(node => (!node.logPw)(topics(node))).sum
      val logZ: Double = dnodes.par.map(_.logP(topics, groups)).sum +
        unodes.par.map(_.logP(topics, groups)).sum
      val logQ: Double = dnodes.par.map(node => log((!node.z)(topics(node)))).sum +
        unodes.par.map(node => log((!node.r)(groups(node)))).sum
      val logP: Double = logPZ + logZ
      logP - logQ
    }.seq
    lse(samples) - log(samples.size)
  }

  ////////////////////////////////////////////////////////////////
  // USER NODE
  ////////////////////////////////////////////////////////////////
  sealed class UNode(val user: Int, val documents: Seq[DNode]) extends TermContainer {

    def dist: Double = norm(!r :- oldR)

    val oldR: Vector = DenseVector.zeros[Double](G)
    val r: Term[Vector] = Term {
      //      val oldR = (!r).copy
//      val n: Vector = DenseVector.zeros[Double](G)
//      n := !logPhi
      val newR: Vector = DenseVector.zeros(G)
      for(c <- 0 until G) {
        val terms: Seq[Double] = documents.map { case (node) =>
          node.parent match {
            case None =>
              lse((!node.logZ) + (!logPi)(c))
            case Some(p) =>
              val o = log((!p.z) * (!node.z).t) + (!logA)(c)
              //if(user == 0) println(o)
              lse(o.toArray)
          }
        }
        newR(c) = (!logPhi)(c) + lse(terms)
      }
      exp(newR - lse(newR))
    }.initialize {
      val n = DenseVector.zeros[Double](G)
      n := normalize(DenseVector.rand[Double](G), 1.0)
      n
    }

    def logP(topics: Map[DNode, Int], groups: Map[UNode, Int]): Double = {
      if(G <= 1)
        0d
      else
        (!logPhi) (groups(this)-1)
    }

    val groupProbs: Term[Map[Int, Double]] = Term {
      (!r).toArray.zipWithIndex.map(t => (t._2, t._1)).toMap
    }

    override def reset(): Unit = {
      oldR := !r
      super.reset()
    }
  }


  ////////////////////////////////////////////////////////////////
  // DOCUMENT NODE
  ////////////////////////////////////////////////////////////////
  class DNode(override val document: Document, override val index: Int)
    extends DocumentNode[DNode](document, index) with TermContainer {

    var author: UNode = _

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

    val lambda: Term[Vector] = Term {
      // Lambda messages regarding likelihood of observing the document
      val msg1 = !logPw
      // Lambda messages from children
      val msg2 = replies.map(!_.lambdaMsg).fold(ZERO)(_ + _)
      msg1 + msg2
    }

    /** Return: Kx1 vector for each value of their parent. **/
    val lambdaMsg: Term[Vector] = Term {
      lse((0 until G).map((g) => lse(a(g), !lambda) * (!author.r)(g)).toArray)
    }

    val tau: Term[Vector] = Term {
      parent match {
        case None =>
          lse((!logPi).zipWithIndex.map { case (pi_g, g) =>
            pi_g * (!author.r) (g)
          })
        case Some(p) =>
          lse((0 until G).map((g) => lse(a(g), !tauMsg) * (!author.r)(g)).toArray)
//          val msg2 = siblings
//            .map((sibling) => !sibling.tauMsg)
//            .fold(DenseVector.zeros[Double](K))(_ + _)
//          msg1 + msg2
      }
    }

    val tauMsg: Term[Vector] = Term {
      parent match {
        case Some(p) => (Array(!p.tau) ++ siblings.map(!_.lambdaMsg)).reduce(_ + _)
        case None => null
      }
    }

    def dist: Double = norm(!z - oldZ)

    val oldZ: Vector = DenseVector.zeros[Double](K)
    val logZ: Term[Vector] = Term {
      val tmp = (!lambda) + !tau
      tmp - lse(tmp)
    }.initialize {
      val a = log(DenseVector.rand[Double](K))
      a - lse(a)
    }
    val z: Term[Vector] = Term { exp(!logZ) }


    val topicProbs: Term[Map[Int, Double]] = Term {
      (!z).toArray.zipWithIndex.map(t => (t._2, t._1)).toMap
    }

    val logLikelihood: Term[Double] = Term {
      parent match {
        case None =>
          lse(pi.zipWithIndex.map { case (piG, g) => lse(!logPw :+ log(piG * (!author.r)(g))) })
        case Some(parent) =>
          lse(a.zipWithIndex.map { case (aG, g) => lse(!logPw :+ log(aG.t * !parent.z * (!author.r)(g)))})
      }
    }

    def logP(topics: Map[DNode, Int], groups: Map[UNode, Int]): Double = {
      val z = topics(this)
      val y = groups(author)
      val term1: Double = parent match {
        case Some(p) =>
          val zp: Int = topics(p)
          (!logA)(y)(zp, z)
        case None =>
          (!logPi)(y)(z)
      }
//      val term2: Double = replies.foldLeft(0d)((p, reply) => p + reply.logP(topics, groups))
      term1
    }

    override def reset(): Unit = {
      oldZ := !z
      super.reset()
    }
  }

  sealed class DTree(val root: DNode) {
    import params._
    private def expand(node: DNode): Seq[DNode] = {
      val children =
        if (node.replies != null) node.replies.flatMap((child) => expand(child))
        else Seq()
      Seq(node) ++ children
    }

    lazy val nodes: Seq[DNode] = expand(root)
    lazy val users: Seq[UNode] = nodes.map(_.author).distinct

    def reset(): Unit = nodes.foreach { (node) => node.reset() }

    def loglikelihood: Double = nodes.map(!_.logLikelihood).sum

    def logP(topics: Map[DNode, Int], groups: Map[UNode, Int]) = {
      nodes.map(_.logP(topics, groups)).sum
    }
  }

  def update(trees: Seq[DTree], dnodes: Seq[DNode], unodes: Seq[UNode], G: Int, numIterations: Int): Unit = {
    import scala.util.control.Breaks._
    val roots = dnodes.filter(_.isRoot)
    breakable {
      for(i <- 1 to numIterations) {
//        println(s" $i")
//        println("  category")
        unodes.par.foreach(author => author.reset())
        unodes.par.foreach(author => author.r.update())
//        println("  nodes")
        dnodes.par.foreach(document => document.reset())
        dnodes.par.foreach(document => document.update())
        val docError: Double = dnodes.map(_.dist).sum / dnodes.size
        val userError: Double = unodes.map(_.dist).sum / unodes.size
//        val docUncert: Double = sum(dnodes.map((dnode) => if(any(!dnode.z :> 0.5)) 1 else 0))
//        val userUncert: Double = sum(unodes.map((unode) => if(any(!unode.r :> 0.5)) 1 else 0))
        if(docError <= 1e-3 && userError <= 1e-3)
          break
      }
    }
  }
}
