package edu.utulsa.conversation.tm

import scala.collection.mutable
import breeze.linalg._
import breeze.numerics.{log, log1p, exp, pow}
import java.io.{File}
import org.json4s._
import org.json4s.native.JsonMethods._
import org.json4s.native.Serialization
import org.json4s.native.Serialization.{write}
import java.io.PrintWriter


class CTopicModel(override val corpus: Corpus)
    extends TopicModel(corpus) with TopicUtils {
  /** MODEL PARAMETERS **/
  var pi: DenseVector[Double]          = null // k x 1
  var logPi: DenseVector[Double]       = null
  var a: DenseMatrix[Double]           = null // k x k
  var logA: DenseMatrix[Double]        = null
  var theta: DenseMatrix[Double]       = null // m x k
  var logTheta: DenseMatrix[Double]    = null
  var scaledTheta: DenseMatrix[Double] = null

  /** LATENT VARIABLE ESTIMATES **/
  private var q: DenseMatrix[Double]     = null // k x n

  /** USEFUL INTERMEDIATES **/
  private var responses: DenseVector[Double] = null // n x 1
                                                    // Reply matrix
  private var b: CSCMatrix[Double]         = null   // n x n
                                                    // Word occurance matrix
  private var c: CSCMatrix[Double]         = null   // n x m

  private def buildResponses() = {
    val N = corpus.numDocuments
    val M = corpus.numWords

    val m = DenseVector.zeros[Double](N)
    corpus.replies.zipWithIndex.foreach{ case (document, index) =>
      m(index) = document.children.length.toDouble
    }

    m
  }

  private def buildB(): CSCMatrix[Double] = {
    val N = corpus.numDocuments
    val M = corpus.numWords

    val builder = new CSCMatrix.Builder[Double](N, N)

    corpus.replies.zipWithIndex.foreach { case (document: Document, source: Int) =>
      document.children.foreach { case (reply: Document) =>
        builder.add(document.index, reply.index, 1d)
      }
    }

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
    println(s" $N documents")
    println(s" $M words")

    println(" Building param matrices...")
    pi = normalize(DenseVector.rand(K))
    logPi = log(pi)
    a = normalize(DenseMatrix.rand(K, K), Axis._1, 1.0)
    logA = log(a)
    theta = normalize(DenseMatrix.rand(M, K), Axis._0, 1.0)
    logTheta = log(theta)

    println(" Building expectation matrix...")
    q = DenseMatrix.zeros[Double](K, N)

    println(" Populating auxillary matrices...")
    responses = buildResponses()
    b = buildB()
    c = buildC()
  }

  def computeScaledTheta(): this.type = {
    val N = corpus.numDocuments
    val M = corpus.numWords
    // scaledTheta = normalize(theta, Axis._1, 1.0)
    val p$z = normalize(sum(q, Axis._1), 1.0)
    scaledTheta = normalize(theta(*, ::) :* p$z, Axis._1, 1.0)
    // // println(s"p(z) ${(p$z.length)}")
    // // scaledTheta = normalize(theta(*,::) :* z, Axis._1, 1.0) :* theta
    // val num = theta(*, ::) :* p$z
    // // println(s"p(theta ^ z) ${(num.rows, num.cols)}")
    // val p$w = sum(num, Axis._1)
    // // println(s"p(w) ${(p$w.length)}")
    // val den = tile(p$z.t, 1, M) :+ tile(p$w, 1, K) :- num
    // // println(s"p(theta v z) ${(den.rows, den.cols)}")
    // scaledTheta = num :/ den
    this
  }

  override def train(): this.type = {
    println("Initializing...")
    initialize()
      (1 to numIterations).foreach { (interval) =>
        println(s" Interval: $interval")
        println("  - E step")
        optimize.expectation(interval)
        println("  - M step")
        optimize.maximization(interval)
      }
    computeScaledTheta()
    this
  }

  // case class WResult(word: String, weight: Double)
  override def save(dir: String): Unit = {
    val d = (if(!dir.endsWith("/")) dir + "/" else dir) + "nctm/"
    new File(d).mkdirs()
    // csvwrite(new File(d + "pi.mat"), pi)
    csvwrite(new File(d + "a.mat"), a)
    csvwrite(new File(d + "theta.mat"), theta)
    csvwrite(new File(d + "scaled_theta.mat"), scaledTheta)
    csvwrite(new File(d + "q.mat"), q)

    implicit val formats = Serialization.formats(NoTypeHints)
    val M = corpus.numWords
    val wordWeights: Map[Int, Map[String, Double]] = (0 until K).map { case (k) =>
      k -> (0 until M).map { case (w) =>
        corpus.dict(w) -> scaledTheta(w, k)
      }.toMap
    }.toMap

    Some(new PrintWriter(d + "word_weights.json"))
      .foreach { (p) => p.write(write(wordWeights)); p.close() }

    corpus.save(d)
  }

  protected object optimize {
    val (trees: Seq[DTree], nodes: Seq[DNode]) = {
      val nodes = corpus.documents.map((document) =>
        new DNode(document, document.index)
      )
      nodes.foreach { (node) =>
        val d = node.document
        if(d.parent != null)
          node.parent = nodes(d.parent.index)
        node.children = d.children.map((c) => nodes(c.index))
      }
      val trees = nodes.filter((node) => node.parent == null)
        .map((node) => new DTree(node))
      (trees, nodes)
    }
    val roots: Seq[Int] = trees.map((tree) => tree.root.index)
    var currentTime = 1

    class Param[T >: Null]() {
      var updated: Int = 0
      var value: T = null
      def apply(): T = value
      def :=(update: => T): Unit = {
        if(currentTime > this.updated) {
          this.updated = currentTime
          this.value = update
        }
      }
    }

    sealed class DNode(val document: Document, val index: Int) extends TopicUtils {
      var parent: DNode = null
      var children: Seq[DNode] = Seq()
      lazy val siblings: Seq[DNode] = {
        if(parent != null) {
          parent.children.filter((child) => child != this)
        }
        else {
          Seq()
        }
      }

      def isRoot: Boolean = this.parent == null

      val pprobW = new Param[DenseVector[Double]]()
      /**
        * Computes log probabilities for observing a set of words for each latent
        * class.
        */
      def probW: DenseVector[Double] = {
        pprobW := {
          val result =
            if(document.words.length > 0)
              document.count
                .map { case (word, count) => logTheta(word, ::).t :* count.toDouble }
                .reduce(_ + _)
            else
              DenseVector.zeros[Double](K)
          result
        }
        pprobW()
      }

      val plambdaMsg = new Param[DenseVector[Double]]()
      def lambdaMsg: DenseVector[Double] = {
        plambdaMsg := { lse(a, this.lambda) }
        plambdaMsg()
      }

      val plambda = new Param[DenseVector[Double]]
      def lambda: DenseVector[Double] = {
        plambda := {
          // Lambda messages regarding likelihood of observing the document
          val msg1 = this.probW
          // Lambda messages from children
          var msg2 = DenseVector.zeros[Double](K)
          if(children.length > 0) {
            msg2 = children
              .map((child) => child.lambdaMsg)
              .reduce(_ + _)
          }
          msg1 + msg2
        }
        plambda()
      }

      val ptau = new Param[DenseVector[Double]]()
      def tau: DenseVector[Double] = {
        ptau := {
          if(parent == null) {
            logPi
          }
          else {
            val msg1: DenseVector[Double] = parent.tau
            val msg2 = siblings
              .map((sibling) => sibling.lambdaMsg)
              .fold(DenseVector.zeros[Double](K))(_ + _)
            msg1 + msg2
          }
        }
        ptau()
      }

      val pq = new Param[DenseVector[Double]]()
      def update(): Unit = {
        pq := { lambda + tau }
        val qi = exp(pq() :- lse(pq()))
        q(::, index) := qi :/ sum(qi)
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

      def logLikelihood(): Double = {
        sum(root.pq())
      }
    }

    def expectation(interval: Int): Unit = {
      currentTime += 1
      trees.par.foreach { case (tree) => tree.update() }
    }

    def maximization(interval: Int): Unit = {
      Seq(
        () => {
          // Pi Maximization
          pi := roots.map((index) => q(::, index)).reduce(_ + _)
          pi := normalize(pi :+ 1e-4, 1.0)
          logPi := log(pi)
        },
        () => {
          // A maximization
          a := normalize((q * b * q.t) :+ 1e-4, Axis._1, 1.0)
          logA := log(a)
        },
        () => {
          // Theta maximization
          theta := normalize((q * c) :+ 1e-8, Axis._1, 1.0).t
          logTheta := log(theta)
        }
      ).par.foreach { case (step) => step() }

      // val count = sum(q.map((item) => if(item > 0.5) 1 else 0))
      // val total = corpus.numDocuments
      // println(f"$count%8d/$total")
      println(s"   ~likelihood: ${logLikelihood()}")
    }

    def logLikelihood(): Double = {
      val p1 = sum(roots.map((index) => q(::, index) :* log(pi)).reduce(_ + _))
      val p2 = sum((q * b * q.t) :* log(a))
      val p3 = sum(
        nodes.map {
          (node) => sum(node.pprobW() :* q(::, node.index))
        }
      )
      p1 + p2 + p3
    }
  }
}

object CTopicModel {
  def apply(corpus: Corpus) = new CTopicModel(corpus)
}
