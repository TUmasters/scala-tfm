package edu.utulsa.conversation.tm

import breeze.linalg._
import breeze.numerics.{exp, log}
import java.io.File

import edu.utulsa.conversation.text.{Corpus, Dictionary, Document, DocumentNode}
import edu.utulsa.conversation.tm.catfm.VariationalEMOptimizer
import edu.utulsa.util.{Term, TermContainer}
import edu.utulsa.util.math._

class LatentCommunityTFM
(
  override val numTopics: Int,
  val numWords: Int,
  val numCommunities: Int,
  val numIterations: Int,
  val maxEIterations: Int
) extends TopicModel(numTopics) with CATFMParams {
  val K: Int = numTopics
  val M: Int = numWords
  val C: Int = numCommunities

  /** MODEL PARAMETERS **/
  val pi: Array[DV]    = (1 to C).map(_ => normalize(DenseVector.rand(K))).toArray // k x g
  val phi: DV          = normalize(DenseVector.rand(C), 1.0) // 1 x g
  val a: Array[DM]     = (1 to C).map(_ => normalize(DenseMatrix.rand(K, K), Axis._1, 1.0)).toArray // g x k x k
  val theta: DM        = normalize(DenseMatrix.rand(M, K), Axis._0, 1.0) // m x k

  private var optim: VariationalEMOptimizer = _

  override lazy val params: Map[String, AnyVal] = super.params ++ Map(
    "num-communities" -> numCommunities,
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
    for(c <- 0 until numCommunities) {
      csvwritevec(new File(dir + f"/pi.g$c%02d.csv"), pi(c))
      csvwrite(new File(dir + f"/a.g$c%02d.csv"), a(c))
    }

    // save user info
    val userGroups: Map[String, List[TPair]] = optim.cnodes.map((node) => {
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
    optim = new VariationalEMOptimizer(corpus, this)
    optim.fit(numIterations, maxEIterations)
  }

  override def logLikelihood(corpus: Corpus): Double = {
    val infer = new VariationalEMOptimizer(corpus, this)
    infer.eStep(100)
    infer.approxLikelihood()
  }
}

trait CATFMParams extends TermContainer {
  val M: Int
  val K: Int
  val C: Int

  val pi: Array[DV]
  val logPi: Term[Array[DV]] = Term {
    pi.map(log(_))
  }
  val phi: DV
  val logPhi: Term[DV] = Term {
    log(phi)
  }
  val logSumPi: Term[DV] = Term {
    val sumPi: DV = pi.zipWithIndex
      .map { case (pi_g, g) => pi_g :* phi(g) }
      .reduce(_ + _)
    log(sumPi)
  }
  val a: Array[DM]
  val logA: Term[Array[DM]] = Term {
    a.map(log(_))
  }
  // Used as a ``normalizing constant'' for the variational inference step of each user's group
  val logSumA: Term[DM] = Term {
    val sumA: DM = a.zipWithIndex
      .map { case (a_g, g) => a_g :* phi(g) }
      .reduce(_ + _)
    log(sumA)
  }
  val theta: DM
  val logTheta: Term[DM] = Term {
    log(theta)
  }
}
