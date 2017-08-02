// package ml.driver

// import ml.topic._
// import org.json4s._
// import org.json4s.native.JsonMethods._
// import org.json4s.native.Serialization
// import org.json4s.native.Serialization.{write}
// import java.io.PrintWriter

// case class Tweet(id: String, words: List[String], replies: List[String], author: String)

// object Twitter {
//   var dataFolder = "data/reddit/politics/"

//   def load(filename: String): List[Tweet] = {
//     println("Loading JSON data...")
//     implicit val formats = DefaultFormats
//     val content = scala.io.Source.fromFile(filename).mkString
//     val json = parse(content)
//     json.extract[List[Tweet]]
//   }

//   def process(comments: Seq[Tweet]): (Corpus, Seq[Seq[Int]], Seq[Seq[Int]]) = {
//     println("Processing data...")
//     println(" - Getting content")
//     val documents = comments
//       .map((comment) => comment.words)
//     println(" - Generating corpus")
//     val corpus = Corpus(documents)
//     println(" - Assigning document IDs")
//     val ids = comments.zipWithIndex
//       .map { case (comment, index) => (comment.id, index) }.toMap
//     println(" - Generating replies")
//     val replies = comments
//       .map((comment) => comment.replies
//         .filter((reply) => ids contains reply)
//         .map((reply) => ids(reply)))
//     println(" - Generating authors")
//     val authors = comments
//       .zipWithIndex
//       .groupBy { case (document: Tweet, index: Int) => document.author }
//       .map { case (k, v: Seq[(Tweet, Int)]) => v.map(_._2).toSeq }
//       .toSeq
//     (corpus, replies, authors)
//   }

//   def train(corpus: Corpus, replies: Seq[Seq[Int]], authors: Seq[Seq[Int]]): UCTopicModel = {
//     println("Training model...")
//     val model = UCTopicModel(corpus, replies, authors)
//       .setNumTopics(40)
//       .setNumIntervals(60)

//     model.train()

//     model
//   }

//   case class TopicData(topic: Int, prob: Double)
//   case class DocumentData(id: String, topics: List[TopicData])
//   def results(comments: Seq[Tweet], corpus: Corpus, replies: Seq[Seq[Int]], model: UCTopicModel): Unit = {
//     // (0 until model.K).foreach((topic) => {
//     //   val pi = model.pi(topic)
//     //   println(f"Topic $topic%3d $pi")
//     //   model.topicDist(topic).likely(10).foreach { case (p, w) => {
//     //     val word = corpus.dictionary(w)
//     //     println(f" - $p%8f : $word")
//     //   }}
//     // })
//     val topics = (comments zip model.documentDists)
//       .map { case (comment, document) =>
//         DocumentData(
//           comment.id,
//           document.dist.take(2).map(a => TopicData(a._1, a._2)).toList
//         ) }
//       .toList

//     implicit val formats = Serialization.formats(NoTypeHints)

//     Some(new PrintWriter(dataFolder + "topics.json"))
//       .foreach { p => p.write(write(topics)); p.close }
//   }

//   def main(args: Array[String]): Unit = {
//     dataFolder = args(0)
//     val comments = load(dataFolder + "tweets.json")

//     val (corpus, replies, authors) = process(comments)

//     val model = train(corpus, replies, authors)
//     model.save(dataFolder + "model")

//     results(comments, corpus, replies, model)
//   }
// }
