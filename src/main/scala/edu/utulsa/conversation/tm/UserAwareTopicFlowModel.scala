package edu.utulsa.conversation.tm

import breeze.linalg._
import breeze.numerics.{exp, log, log1p, pow}
import java.io.File

import org.json4s._
import org.json4s.native.JsonMethods._
import org.json4s.native.Serialization
import org.json4s.native.Serialization.write
import java.io.PrintWriter

import edu.utulsa.conversation.text.{Corpus, Document}

class UCTMParams(
  val pi: Array[DenseVector[Double]],
  val a: Array[DenseMatrix[Double]],
  val theta: DenseMatrix[Double]) extends MathUtils {

  val id: Int = rand.nextInt()

  /** Track these values just in case we need them **/
  lazy val logPi: Array[DenseVector[Double]] = pi.map(pig => log(pig))
  lazy val logA: Array[DenseMatrix[Double]] = a.map(ag => log(ag))
  lazy val logTheta: DenseMatrix[Double] = log(theta)
}

/**
  * Basically a class for memoization, that updates based on cached value.
  */
sealed class Term[T >: Null](update: (UCTMParams) => T) {
  private var updated: Int = 0
  private var value: T = null
  def apply()(implicit params: UCTMParams): T = {
    if(params.id != this.updated) {
      this.updated = params.id
      this.value = this.update(params)
    }
    this.value
  }
  def get: T = this.value
}

object Term {
  def apply[T >: Null](update: (UCTMParams) => T): Term[T] = {
    new Term[T](update)
  }
}


sealed class InferenceSolver(val q: DenseMatrix[Double], val r: DenseMatrix[Double], val qr: Array[DenseMatrix[Double]], val K: Int, val G: Int)(corpus: Corpus) {
  private def collect(corpus: Corpus): (Seq[DTree], Seq[DNode], Seq[UNode]) = {

    val dnodes: Map[Document, DNode] = corpus.documents.zipWithIndex.map { case (document, index) =>
      document -> new DNode(document, index)
    }.toMap

    println(" - Creating author nodes...")
    val unodes: Map[Int, UNode] = corpus.authors.items.map { case (author: Int) =>
      author -> new UNode(author)
    }.toMap

    println(" - Mapping replies...")
    dnodes.foreach { case (document, node) =>
      if(document.parent != null)
        node.parent = dnodes(document.parent)
      node.children = document.replies.map(dnodes(_)).toSeq
    }

    println(" - Mapping users to authors...")
    corpus.documents.groupBy(_.author).foreach { case (author: Int, documents: List[Document]) =>
      unodes(author).documents = documents.map(dnodes(_)).toSeq
      documents.foreach((document) =>
        dnodes(document).author = author
      )
    }

    println(" - Creating trees...")
    val trees = dnodes.map(_._2)
      .filter((node) => node.isRoot)
      .map((root) => new DTree(root))
      .toSeq

    (trees, dnodes.map(_._2).toSeq, unodes.map(_._2).toSeq)
  }

  val (trees: Seq[DTree], dnodes: Seq[DNode], unodes: Seq[UNode]) = collect(corpus)
  lazy val roots = trees.map((tree) => tree.root.index)

  implicit var params: UCTMParams = null

  def update(pi: Array[DenseVector[Double]],
    a: Array[DenseMatrix[Double]],
    theta: DenseMatrix[Double]): Unit = {

    import scala.util.control.Breaks._

    breakable {
      for(i <- 1 to 10) {
        println(s"    - Interval $i")
        // println(q(::, 1))
        // println(r(::, 1))
        this.params = new UCTMParams(pi, a, theta)
        // println(q(::, 1))
        println("      * document update")
        trees.par.foreach  (_.update())
        // println(r(::, 1))
        println("      * user update")
        unodes.par.foreach (_.update())
        val derror = dnodes.map((dnode) => dnode.dist).reduce(_ + _) / dnodes.size
        val dncert = sum(dnodes.map((dnode) => if(any(dnode.q_d.get :> 0.5)) 1 else 0))
        val uerror = unodes.map((unode) => unode.dist).reduce(_ + _) / unodes.size
        val uncert = sum(unodes.map((unode) => if(any(unode.r_g.get :> 0.5)) 1 else 0))
        println(f"      error: document $derror%4e user $uerror%4e")
        println(f"      count: document $dncert%6d/${q.cols}%6d user $uncert%6d/${r.cols}%6d")
        println(r(::, 0))
        if(derror <= 1e-4 && uerror <= 1e-4)
          break
      }
    }

    (0 until G).foreach { (g) =>
      qr(g) := q
      unodes.foreach { (unode) =>
        unode.documents.foreach { (dnode) =>
          qr(g)(::, dnode.index) :*= unode.r_g()(params)(g)
        }
      }
    }
  }

  ////////////////////////////////////////////////////////////////
  // USER NODE
  ////////////////////////////////////////////////////////////////
  sealed class UNode(val user: Int) extends MathUtils {
    var documents: Seq[DNode] = Seq()
    val r_g = Term[DenseVector[Double]] { (params) =>
      implicit val tmp = params
      val n = documents.map { case (node) =>
        if(node.isRoot)
          DenseVector[Double]((0 until G).map((g) =>
            log(params.pi(g) dot node.q_d())
          ).toArray)
        else
          DenseVector[Double]((0 until G).map((g) =>
            log(node.parent.q_d().t * params.a(g) * node.q_d())
          ).toArray)
      }.reduce(_ + _)
      // Normalize
      exp(n :- lse(n))
    }

    var dist: Double = G

    def update() = {
      val old_r = r_g.get
      r(::, user) := r_g()
      if(old_r != null)
        dist = norm(old_r - r_g())
    }
  }

  sealed class DNode(val document: Document, val index: Int) extends MathUtils {
    var parent: DNode = null
    var author: Int = -1
    var children: Seq[DNode] = Seq()
    lazy val siblings: Seq[DNode] = {
      if(parent != null) {
        parent.children.filter((child) => child != this)
      }
      else {
        Seq()
      }
    }

    def isRoot: Boolean = { this.parent == null }

    /**
      * Computes log probabilities for observing a set of words for each latent
      * class.
      */
    val probW = Term[DenseVector[Double]] { (params) =>
      implicit val tmp = params
      val result =
        if(document.words.length > 0)
          document.count
            .map { case (word, count) => params.logTheta(word, ::).t :* count.toDouble }
            .reduce(_ + _)
        else
          DenseVector.zeros[Double](K)
      result
    }

    val lambdaMsg = Term[DenseVector[Double]] { (params) =>
      implicit val tmpparams = params
      lse((0 until G).map((g) => lse(params.a(g), lambda()) :+ log(r(g, author))).toArray)
    }

    val lambda: Term[DenseVector[Double]] = Term[DenseVector[Double]] { (params) =>
      implicit val tmp = params
      // Lambda messages regarding likelihood of observing the document
      val msg1 = probW()
      // Lambda messages from children
      val msg2 =
        if(children.length > 0)
          children
            .map((child) => child.lambdaMsg())
            .reduce(_ + _)
        else
          DenseVector.zeros[Double](K)
      msg1 + msg2
    }

    val pi: Term[DenseVector[Double]] = Term[DenseVector[Double]] { (params) =>
      implicit val tmp = params
      if(parent == null) {
        lse(params.logPi.zipWithIndex.map { case (pi_g, g) =>
          pi_g :+ log(r(g, author))
        })
      }
      else {
        val msg1: DenseVector[Double] = parent.pi()
        val msg2 =
          if(siblings.length > 0) siblings.map((sibling) => sibling.lambdaMsg()).reduce(_ + _)
          else DenseVector.zeros[Double](K)
        msg1 + msg2
      }
    }

    val q_d = Term[DenseVector[Double]] { (params) =>
      implicit val tmpparams = params
      val tmp = lambda() :+ pi()
      exp(tmp :- lse(tmp))
    }

    var dist: Double = K

    def update(): Unit = {
      val old_q = q_d.get
      q(::, index) := q_d()
      if(old_q != null)
        dist = norm(old_q - q_d())
    }
  }

  sealed class DTree(val root: DNode) {
    private def expand(node: DNode): Seq[DNode] = {
      val children =
        if(node.children != null) node.children.map((child) => expand(child)).flatten
        else Seq()
      Seq(node) ++ children
    }
    lazy val nodes: Seq[DNode] = expand(root)

    def update(): Unit = {
      nodes.foreach { (node) => node.update() }
    }
  }
}

class UCTopicModel(override val corpus: Corpus)
    extends TopicModel(corpus) with MathUtils {
  var G: Int = 5
  def setNumUserGroups(numGroups: Int): this.type = {
    this.G = numGroups
    this
  }

  /** MODEL PARAMETERS **/
  var pi: Array[DenseVector[Double]]    = null // k x g
  var a: Array[DenseMatrix[Double]]     = null // g x k x k
  var theta: DenseMatrix[Double]        = null // m x k
  var scaledTheta: DenseMatrix[Double]  = null

  /** LATENT VARIABLE ESTIMATES **/
  private var q: DenseMatrix[Double]     = null // k x n
  private var r: DenseMatrix[Double]     = null // g x u
  private var qr: Array[DenseMatrix[Double]] = null // g x k x n

  /** USEFUL INTERMEDIATES **/
  private var responses: DenseVector[Double] = null // n x 1
  // Reply matrix
  private var b: CSCMatrix[Double]         = null   // n x n
  // Word occurance matrix
  private var c: CSCMatrix[Double]         = null   // n x m

  private var inference: InferenceSolver = null
  private var roots: Seq[Int] = null

  private def buildResponses() = {
    val N = corpus.numDocuments
    val M = corpus.numWords

    val m = DenseVector.zeros[Double](N)
    inference.dnodes.foreach{ case (node) =>
      m(node.index) = node.children.length.toDouble
    }

    m
  }

  private def buildB(): CSCMatrix[Double] = {
    val N = corpus.numDocuments
    val M = corpus.numWords

    val builder = new CSCMatrix.Builder[Double](N, N)

    inference.dnodes.foreach((node) =>
      node.children.foreach((child) =>
        builder.add(node.index, child.index, 1d)
      )
    )

    builder.result()
  }

  private def buildC(): CSCMatrix[Double] = {
    val N = corpus.numDocuments
    val M = corpus.numWords

    val builder = new CSCMatrix.Builder[Double](N, M)

    corpus.documents.zipWithIndex.foreach { case (document, index) =>
      document.count.foreach { case (word, count) =>
        builder.add(index, word, count.toDouble)
      }
    }

    builder.result()
  }

  protected def initialize(): Unit = {
    val N = corpus.numDocuments
    val M = corpus.numWords
    val U = corpus.numUsers
    println(s" $N documents")
    println(s" $M words")
    println(s" $U authors")

    println(" Building param matrices...")
    pi = (1 to G).map((i) => normalize(DenseVector.rand(K))).toArray
    a = (1 to G).map((i) => normalize(DenseMatrix.rand(K, K), Axis._1, 1.0)).toArray
    theta = normalize(DenseMatrix.rand(M, K), Axis._0, 1.0)

    println(" Building expectation matrix...")
    q = normalize(DenseMatrix.rand[Double](K, N), Axis._1, 1.0)
    r = normalize(DenseMatrix.rand[Double](G, U), Axis._1, 1.0)
    qr = (1 to G).map((i) => DenseMatrix.zeros[Double](K, N)).toArray

    inference = new InferenceSolver(q, r, qr, K, G)(corpus)

    println(" Populating auxillary matrices...")
    responses = buildResponses()
    b = buildB()
    c = buildC()

    roots = inference.roots
  }

  protected def expectation(interval: Int): Unit = {
    inference.update(pi, a, theta)
  }

  protected def maximization(interval: Int): Unit = {
    Seq(
      () => {
        // Pi Maximization
        pi.zipWithIndex.foreach { case (pi_g, g) =>
          pi_g := normalize(roots.map((index) => qr(g)(::, index)).reduce(_ + _) :+ 1e-6, 1.0)
        }
      },
      () => {
        // A maximization
        a.zipWithIndex.foreach { case (a_g, g) =>
          a_g := normalize((q * b * qr(g).t) :+ 1e-6, Axis._1, 1.0)
        }
      },
      () => {
        // Theta maximization
        theta := normalize((q * c) :+ 1e-8, Axis._1, 1.0).t
      }
    ).par.foreach { case (step) => step() }

    // println(s"   ~likelihood: ${inference.logLikelihood()}")
  }

  def computeScaledTheta(): this.type = {
    val N = corpus.numDocuments
    val M = corpus.numWords
    val p$z = normalize(sum(q, Axis._1), 1.0)
    scaledTheta = normalize(theta(*, ::) :* p$z, Axis._1, 1.0)
    this
  }

  override def train(): this.type = {
    println("Initializing...")
    initialize()
    (1 to numIterations).foreach { (interval) =>
      println(s" Interval: $interval")
      println("  - E step")
      expectation(interval)
      println("  - M step")
      maximization(interval)
    }
    computeScaledTheta()
    this
  }

  sealed case class UResult(id: Int, numPosts: Int, group: (Double, Int))
  override def save(dir: String): Unit = {
    val d = (if(!dir.endsWith("/")) dir + "/" else dir) + "uctm/"
    new File(d).mkdirs()

    (0 until G).foreach { (g) => csvwrite(new File(d + s"a_$g.mat"), a(g)) }
    csvwrite(new File(d + "theta.mat"), theta)
    csvwrite(new File(d + "scaled_theta.mat"), scaledTheta)
    csvwrite(new File(d + "q.mat"), q)

    implicit val formats = Serialization.formats(NoTypeHints)
    val M = corpus.numWords
    val wordWeights: Map[Int, Map[String, Double]] = (0 until K).map { case (k) =>
      k -> (0 until M).map { case (w) =>
        // WordWeight(corpus.dictionary(w), scaledTheta(w, k))
        corpus.words(w) -> scaledTheta(w, k)
      }.toMap
    }.toMap
    Some(new PrintWriter(d + "word_weights.json"))
      .foreach { (p) => p.write(write(wordWeights)); p.close() }

    val users: List[UResult] = inference.unodes.map((unode) =>
      UResult(unode.user, unode.documents.length, unode.r_g.get.toArray.zipWithIndex.maxBy(_._1))
    ).toList
    Some(new PrintWriter(d + "users.json"))
      .foreach { (p) => p.write(write(users)); p.close() }

    corpus.save(d)
  }
}

object UserAwareTopicFlowModel {
  def apply(corpus: Corpus) = new UCTopicModel(corpus)
}
