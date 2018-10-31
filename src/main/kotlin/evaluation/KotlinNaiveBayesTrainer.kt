package evaluation

import edu.unh.cs753.BayesCounter
import edu.unh.cs753.utils.parsing.EmailParsing
import org.apache.commons.io.IOUtils
import org.apache.lucene.analysis.Analyzer
import org.apache.lucene.analysis.en.EnglishAnalyzer
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute
import tech.blueglacier.email.Email
import utils.KotlinDataUtils
import java.io.StringReader


// Temporary class until the group figures things out...
data class KotlinEmail(
        val subject: String = "",
        val text: String = "",
        val label: String = "",
        val tokens: List<String> = emptyList(),
        val emailId: String = "") {

    companion object {

        public fun readEmail(emailLoc: String, label: String): KotlinEmail? =
                try { readEmailHelper(emailLoc, label) } catch (e: Exception) { null }
//        readEmailHelper(emailLoc, label)

        private fun readEmailHelper(emailLoc: String, label: String): KotlinEmail {
            val email = EmailParsing.readMail(emailLoc)
            val subject = email.getEmailSubject() ?: ""
            val content = getContent(email)  ?: ""
            if (content == "")
                println(emailLoc)
            val tokens = createTokenList(content, EnglishAnalyzer())
            val emailId = emailLoc.split("/").last()

            return KotlinEmail(
                    subject = subject,
                    text = content,
                    label = label,
                    tokens = tokens,
                    emailId = emailId
            )
        }

        fun getContent(email: Email): String? {
            return if (email.plainTextEmailBody != null) {
                email.plainTextEmailBody.`is`.bufferedReader().readText()
            } else {
                val html = IOUtils.toString(email.getHTMLEmailBody().getIs())
                EmailParsing.convertHtmlToPlainText(html)
            }
        }
    }
}

class KotlinNaiveBayesTrainer() {
    val bayesCounter = BayesCounter()

    fun doTrain(tsvLoc: String) {
        val emails = KotlinEmailParser.readEmailTsv(tsvLoc)
        val (train, test) = KotlinEmailParser.createTestTrainData(emails, 0.5)

        train.forEach { email ->
            bayesCounter.buildHashMap(email.label, email.tokens)
        }

        val calledLabels = test.map { email ->
            email.emailId to bayesCounter.classify(email.tokens)
        }.toMap()

        val trueLabels = test.map { it.emailId to it.label }
            .toMap()

        val evaluator = KotlinEvaluator(trueLabels)
        val f1 = evaluator.evaluateCalledLabelsUsingF1(calledLabels)
        println("F1 Score for Naive Bayes: $f1")
    }

}


private fun createTokenList(text: String, analyzer: Analyzer): ArrayList<String> {
    val tokens = ArrayList<String>()

    val stringReader = StringReader(text)
    val ts = analyzer.tokenStream("text", stringReader)
    ts.reset()
    while (ts.incrementToken()) {
        tokens.add(ts.getAttribute(CharTermAttribute::class.java).toString())
    }
    ts.end()
    ts.clearAttributes()

    return tokens
}

fun main(args: Array<String>) {
    val spamLoc = "/home/hcgs/data_science/data/spam/trec07p"
    val tsvLoc = "/home/hcgs/Desktop/projects/assignments/team2_group_project/parsed_emails.tsv"
    val emails = KotlinEmailParser.readEmailTsv(tsvLoc)
    val (train, test) = KotlinEmailParser.createTestTrainData(emails, 0.5)

    val trainer = KotlinNaiveBayesTrainer()
//    trainer.doTrain(train, test)
}