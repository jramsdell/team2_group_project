package evaluation

import edu.unh.cs753.predictors.LabelPredictor
import utils.pmap
import java.io.File
import java.util.*


class KotlinEvaluator( val correctLabels: Map<String, String> ) {

    fun evaluateCalledLabelsUsingF1(calledLabels: Map<String, String>): Double {
        var tn = 0.0
        var tp = 0.0
        var fn = 0.0
        var fp = 0.0

        calledLabels.forEach { (id, label) ->
            val correctLabel = correctLabels[id]!!
//            println("$correctLabel\t$label")
            val isSpam = correctLabel == "spam"

//            if (correctLabel == "spam" && label == "spam") { tp += 1.0 }
//            else if (correctLabel == "spam" && label == "ham") { fn += 1.0 }
//            else if (correctLabel == "ham" && label == "ham") { tn += 1.0 }
//            else { fp += 1.0 }

            if (isSpam && label == "spam") { tp += 1 }
            else if (isSpam && label != "spam") { fn += 1 }
            else if (!isSpam && label == "ham" ) { tn += 1 }
            else { fp += 1 }

        }


        val precision = tp.toDouble() / (tp + fp)
        val recall = tp.toDouble() / (tp + fn)
        val f1 = 2 * (precision * recall) / (precision + recall)
        val precision2 = tn.toDouble() / (tn + fn)
        val recall2 = tn.toDouble() / (tn + fp)
        val f2 = 2 * (precision2 * recall2) / (precision2 + recall2)
        return (f1 + f2)/2
    }

    companion object {
        fun extractLabels(labelsFile: String) =
            File(labelsFile)
                .bufferedReader()
                .readLines()
                .map { line -> line.split(" ").let { it[0] to it[1] } }
                .toMap()


        fun evaluate(lp: LabelPredictor, nEvals: Int = -1)  {
            val emails = KotlinEmailParser.readEmailTsv("parsed_emails.tsv")
            val test = kotlin.run {
                val (_, testy) = KotlinEmailParser.createTestTrainData(emails, 0.5)
                if (nEvals == -1) testy else testy.shuffled(Random(123)).take(nEvals)
            }



            val calledLabels = test.map { email ->
                email.emailId to lp.predict(email.tokens)
            }.toMap()

            val trueLabels = test.map { it.emailId to it.label }
                .toMap()

            val evaluator = KotlinEvaluator(trueLabels)
            val f1 = evaluator.evaluateCalledLabelsUsingF1(calledLabels)
            println("F1 Score for Label Prediction Method: $f1")

        }




        fun writeTrainTest()  {
            val emails = KotlinEmailParser.readEmailTsv("parsed_emails.tsv")
            val (train, test) = KotlinEmailParser.createTestTrainData(emails, 0.5)

            val trainOut = File("train_emails.tsv").bufferedWriter()
            val testOut = File("test_emails.tsv").bufferedWriter()

            train.map { "${it.emailId}\t${it.label}\t${it.tokens.joinToString(" ")}" }
                .joinToString("\n")
                .run { trainOut.write(this) }

            test.map { "${it.emailId}\t${it.label}\t${it.tokens.joinToString(" ")}" }
                .joinToString("\n")
                .run { testOut.write(this) }

            trainOut.close()
            testOut.close()

        }


    }

}

fun main(args: Array<String>) {
    KotlinEvaluator.writeTrainTest()
}
