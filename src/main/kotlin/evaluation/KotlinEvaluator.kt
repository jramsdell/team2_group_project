package evaluation

import edu.unh.cs753.predictors.LabelPredictor
import java.io.File


class KotlinEvaluator( val correctLabels: Map<String, String> ) {

    fun evaluateCalledLabelsUsingF1(calledLabels: Map<String, String>): Double {
        var tn = 0
        var tp = 0
        var fn = 0
        var fp = 0

        calledLabels.forEach { (id, label) ->
            val correctLabel = correctLabels[id]!!
//            println("$correctLabel\t$label")
            val isSpam = correctLabel == "spam"

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


        fun evaluate(lp: LabelPredictor)  {
            val emails = KotlinEmailParser.readEmailTsv("parsed_emails.tsv")
            val (_, test) = KotlinEmailParser.createTestTrainData(emails, 0.5)


            val calledLabels = test.map { email ->
                email.emailId to lp.predict(email.tokens)
            }.toMap()

            val trueLabels = test.map { it.emailId to it.label }
                .toMap()

            val evaluator = KotlinEvaluator(trueLabels)
            val f1 = evaluator.evaluateCalledLabelsUsingF1(calledLabels)
            println("F1 Score for Label Prediction Method: $f1")

        }

    }


}
