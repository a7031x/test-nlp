import java.util.Properties

import edu.stanford.nlp.ling.CoreAnnotations
import edu.stanford.nlp.ling.CoreAnnotations._
import edu.stanford.nlp.pipeline.{Annotation, StanfordCoreNLP}
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations

import scala.collection.JavaConversions._

object MyApp extends App {

  def testEnglishSentiment(): Unit = {
    println("======================test sentiment======================")
    val props = new Properties()
    props.setProperty("annotators", "tokenize, ssplit, pos, lemma, parse, sentiment")
    val pipeline = new StanfordCoreNLP(props)
    val annotation = pipeline.process("I don't like the movie. But my friend likes it.")

    for(sentence <- annotation.get(classOf[CoreAnnotations.SentencesAnnotation])) {
      val sentiment = sentence.get(classOf[SentimentCoreAnnotations.SentimentClass])
      println(s"$sentence --- $sentiment")
    }
  }

  def testChineseNER(): Unit = {
    println("======================test Chinese named entity recognition======================")
    val pipeline = new StanfordCoreNLP("chinese.properties")
    val document = new Annotation("克林顿说，华盛顿将逐步落实对韩国的经济援助。金大中对克林顿的讲话报以掌声：克林顿总统在会谈中重申，他坚定地支持韩国摆脱经济危机。")
    pipeline.annotate(document)
    for (sentence <- document.get(classOf[SentencesAnnotation])) {
      for(token <- sentence.get(classOf[TokensAnnotation])) {
        val word = token.get(classOf[TextAnnotation])
        val pos = token.get(classOf[PartOfSpeechAnnotation])
        val ne = token.get(classOf[NamedEntityTagAnnotation])
        println(s"$word: $pos, $ne")
      }
    }
  }
  testChineseNER()
  testEnglishSentiment()
}
