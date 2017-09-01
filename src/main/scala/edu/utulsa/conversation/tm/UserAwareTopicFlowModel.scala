package edu.utulsa.conversation.tm

import breeze.linalg._
import breeze.numerics.{exp, log, log1p, pow}
import java.io.File

import org.json4s._
import org.json4s.native.JsonMethods._
import org.json4s.native.Serialization
import org.json4s.native.Serialization.write
import java.io.PrintWriter

import edu.utulsa.conversation.text.{Corpus, Dictionary, Document}

class UserAwareTopicFlowModel
(
  override val numTopics: Int,
  val numUserGroups: Int,
  override val words: Dictionary,
  val authors: Dictionary,
  override val documentInfo: Map[String, List[TPair]],
  override val wordInfo: Map[String, List[TPair]],
  val pi: Array[DenseVector[Double]],
  val a: Array[DenseMatrix[Double]],
  val theta: DenseMatrix[Double]
) extends TopicModel(numTopics, words, documentInfo, wordInfo) {
  override protected def saveModel(dir: File): Unit = {
    csvwrite(new File(dir + "/theta.csv"), theta)
    for(i <- 0 until numUserGroups) {
      MathUtils.csvwritevec(new File(dir + f"/pi.g$i%02d.csv"), pi(i))
      csvwrite(new File(dir + f"/a.g$i%02d.csv"), a(i))
    }
  }
  override def likelihood(corpus: Corpus): Double = ???
}

object UserAwareTopicFlowModel {
  def train
  (
    corpus: Corpus,
    numTopics: Int,
    numUserGroups: Int,
    numIterations: Int,
    maxEIterations: Int
  ): UserAwareTopicFlowModel = {
    new UATFMOptimizer(corpus, numTopics, numUserGroups, numIterations, maxEIterations)
      .train()
  }
}

sealed class UATFMOptimizer
(
  override val corpus: Corpus,
  override val numTopics: Int,
  val numUserGroups: Int,
  val numIterations: Int,
  val maxEIterations: Int
) extends TMOptimizer[UserAwareTopicFlowModel](corpus, numTopics) {

  import MathUtils._

  override def train(): UserAwareTopicFlowModel = {
    println("Initializing...")
    (1 to numIterations).foreach { (interval) =>
      println(s" Interval: $interval")
      println("  - E step")
      eStep(interval)
      println("  - M step")
      mStep(interval)
    }
    val d: Map[String, List[TPair]] = dnodes.map((node) =>
      node.document.id ->
        node.q_d().data.zipWithIndex.map { case (p, i) => TPair(p, i) }.sortBy(-_.p).toList
    ).toMap
    val w: Map[String, List[TPair]] = (0 until M).map((w) =>
      corpus.words(w) ->
        theta(w, ::).t.toArray.zipWithIndex.map { case (p, i) => TPair(p, i) }.sortBy(-_.p).toList
    ).toMap
    new UserAwareTopicFlowModel(numTopics, numUserGroups, corpus.words, corpus.authors, d, w, pi, a, theta)
  }

  val N: Int = corpus.size
  val M: Int = corpus.words.size
  val U: Int = corpus.authors.size
  def G: Int = numUserGroups

  val (trees: Seq[DTree], dnodes: Seq[DNode], unodes: Seq[UNode]) = {
    val dnodes: Map[Document, DNode] = corpus.zipWithIndex.map { case (document, index) =>
      document -> new DNode(document, index)
    }.toMap

    println(" - Creating author nodes...")
    val unodes: Map[Int, UNode] = corpus.authors.ids.map { case (author: Int) =>
      author -> new UNode(author)
    }.toMap

    println(" - Mapping replies...")
    dnodes.foreach { case (document, node) =>
      node.parent = document.parent match {
        case Some(key) => dnodes(key)
        case None => null
      }
      node.children = document.replies.map(dnodes(_)).toSeq
    }

    println(" - Mapping users to authors...")
    corpus.groupBy(_.author).foreach { case (author: Int, documents: List[Document]) =>
      unodes(author).documents = documents.map(dnodes(_)).toSeq
      documents.foreach((document) =>
        dnodes(document).author = author
      )
    }

    println(" - Creating trees...")
    val trees = dnodes.values
      .filter((node) => node.isRoot)
      .map((root) => new DTree(root))
      .toSeq

    (trees, dnodes.values.toSeq, unodes.values.toSeq)
  }

  class UATFMParams
  (
    val pi: Array[DenseVector[Double]],
    val a: Array[DenseMatrix[Double]],
    val theta: DenseMatrix[Double]
  ) {
    import MathUtils._

    val id: Int = rand.nextInt()

    /** Track these values just in case we need them **/
    lazy val logPi: Array[DenseVector[Double]] = pi.map(pig => log(pig))
    lazy val logA: Array[DenseMatrix[Double]] = a.map(ag => log(ag))
    lazy val logTheta: DenseMatrix[Double] = log(theta)
  }

  /**
    * Basically a class for memoization, that updates based on cached value.
    */
  class Term[T >: Null](update: (UATFMParams) => T) {
    private var updated: Int = 0
    private var value: T = null
    def apply()(implicit params: UATFMParams): T = {
      if(params.id != this.updated) {
        this.updated = params.id
        this.value = this.update(params)
      }
      this.value
    }
    def get: T = this.value
  }

  object Term {
    def apply[T >: Null](update: (UATFMParams) => T): Term[T] = {
      new Term[T](update)
    }
  }

  val roots: Seq[Int] = trees.map(_.root.index)

  implicit var params: UATFMParams = _

  /** MODEL PARAMETERS **/
  val pi: Array[DenseVector[Double]]    = (1 to G).map((i) => normalize(DenseVector.rand(K))).toArray // k x g
  val a: Array[DenseMatrix[Double]]     = (1 to G).map((i) => normalize(DenseMatrix.rand(K, K), Axis._1, 1.0)).toArray // g x k x k
  val theta: DenseMatrix[Double]        = normalize(DenseMatrix.rand(M, K), Axis._0, 1.0) // m x k

  /** LATENT VARIABLE ESTIMATES **/
  val q: DenseMatrix[Double]     = normalize(DenseMatrix.rand[Double](K, N), Axis._1, 1.0) // k x n
  val r: DenseMatrix[Double]     = normalize(DenseMatrix.rand[Double](G, U), Axis._1, 1.0) // g x u
  val qr: Array[DenseMatrix[Double]] = (1 to G).map((i) => DenseMatrix.zeros[Double](K, N)).toArray // g x k x n

  /** USEFUL INTERMEDIATES **/
  val responses: DenseVector[Double] = {
    val m = DenseVector.zeros[Double](N)
    dnodes.foreach{ case (node) =>
      m(node.index) = node.children.length.toDouble
    }

    m
  } // n x 1
  // Reply matrix
  val b: CSCMatrix[Double] = {
    val builder = new CSCMatrix.Builder[Double](N, N)

    dnodes.foreach((node) =>
      node.children.foreach((child) =>
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

  protected def eStep(interval: Int): Unit = {
    import scala.util.control.Breaks._

    breakable {
      for(i <- 1 to 10) {
        println(s"    - Interval $i")
        // println(q(::, 1))
        // println(r(::, 1))
        this.params = new UATFMParams(pi, a, theta)
        // println(q(::, 1))
        println("      * document update")
        trees.par.foreach  (_.update())
        // println(r(::, 1))
        println("      * user update")
        unodes.par.foreach (_.update())
        val derror = dnodes.map((dnode) => dnode.dist).sum / dnodes.size
        val dncert = sum(dnodes.map((dnode) => if(any(dnode.q_d.get :> 0.5)) 1 else 0))
        val uerror = unodes.map((unode) => unode.dist).sum / unodes.size
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

  protected def mStep(interval: Int): Unit = {
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

  ////////////////////////////////////////////////////////////////
  // USER NODE
  ////////////////////////////////////////////////////////////////
  class UNode(val user: Int) {
    var documents: Seq[DNode] = Seq()
    val r_g: Term[DenseVector[Double]] = Term[DenseVector[Double]] { (params) =>
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

    def update(): Unit = {
      val old_r = r_g.get
      r(::, user) := r_g()
      if(old_r != null)
        dist = norm(old_r - r_g())
    }
  }


  ////////////////////////////////////////////////////////////////
  // DOCUMENT NODE
  ////////////////////////////////////////////////////////////////
  class DNode(val document: Document, val index: Int) {
    var parent: DNode = _
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
    val probW: Term[DenseVector[Double]] = Term[DenseVector[Double]] { (params) =>
      implicit val tmp: UATFMParams = params
      val result =
        if(document.words.length > 0)
          document.count
            .map { case (word, count) => params.logTheta(word, ::).t :* count.toDouble }
            .reduce(_ + _)
        else
          DenseVector.zeros[Double](K)
      result
    }

    val lambdaMsg: Term[DenseVector[Double]] = Term[DenseVector[Double]] { (params) =>
      implicit val tmpparams: UATFMParams = params
      lse((0 until G).map((g) => lse(params.a(g), lambda()) :+ log(r(g, author))).toArray)
    }

    val lambda: Term[DenseVector[Double]] = Term[DenseVector[Double]] { (params) =>
      implicit val tmp: UATFMParams = params
      // Lambda messages regarding likelihood of observing the document
      val msg1 = probW()
      // Lambda messages from children
      val msg2 =
        if(children.nonEmpty)
          children
            .map((child) => child.lambdaMsg())
            .reduce(_ + _)
        else
          DenseVector.zeros[Double](K)
      msg1 + msg2
    }

    val pi: Term[DenseVector[Double]] = Term[DenseVector[Double]] { (params) =>
      implicit val tmp: UATFMParams = params
      if(parent == null) {
        lse(params.logPi.zipWithIndex.map { case (pi_g, g) =>
          pi_g :+ log(r(g, author))
        })
      }
      else {
        val msg1: DenseVector[Double] = parent.pi()
        val msg2 =
          if(siblings.nonEmpty) siblings.map((sibling) => sibling.lambdaMsg()).reduce(_ + _)
          else DenseVector.zeros[Double](K)
        msg1 + msg2
      }
    }

    val q_d: Term[DenseVector[Double]] = Term[DenseVector[Double]] { (params) =>
      implicit val tmpparams: UATFMParams = params
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

  class DTree(val root: DNode) {
    private def expand(node: DNode): Seq[DNode] = {
      val children =
        if(node.children != null) node.children.flatMap((child) => expand(child))
        else Seq()
      Seq(node) ++ children
    }
    lazy val nodes: Seq[DNode] = expand(root)

    def update(): Unit = {
      nodes.foreach { (node) => node.update() }
    }
  }

}
