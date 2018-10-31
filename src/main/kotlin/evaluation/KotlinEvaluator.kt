package evaluation

import java.io.File


class KotlinEvaluator( val correctLabels: Map<String, String> ) {

    fun evaluateCalledLabelsUsingF1(calledLabels: Map<String, String>): Double {
        var tn = 0
        var tp = 0
        var fn = 0
        var fp = 0

        calledLabels.forEach { (id, label) ->
            val correctLabel = correctLabels[id]!!
            println("$correctLabel\t$label")
            val isSpam = correctLabel == "spam"

            if (isSpam && label == "spam") { tp += 1 }
            else if (isSpam && label != "spam") { fn += 1 }
            else if (!isSpam && label == "ham" ) { tn += 1 }
            else { fp += 1 }
        }

        val precision = tp.toDouble() / (tp + fp)
        val recall = tp.toDouble() / (tp + fn)
        val f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    }

    companion object {
        fun extractLabels(labelsFile: String) =
            File(labelsFile)
                .bufferedReader()
                .readLines()
                .map { line -> line.split(" ").let { it[0] to it[1] } }
                .toMap()
    }


}
