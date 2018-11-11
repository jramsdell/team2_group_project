package components

import containers.EmailSparseVector
import kernels.SimilarityFuns
import learning.stochastic.StochasticDescent

class StochasticComponent(val basisVectors: List<EmailSparseVector>,
                          trainingVectors: List<EmailSparseVector>,
                          val trainingVectorComponent: TrainingVectorComponent) {
    val spamVectors = trainingVectors.filter { it.label == "spam" }
    val hamVectors = trainingVectors.filter { it.label == "ham" }



    fun getAverageDist(weights: List<Double>): Double {
        val vBad = spamVectors.map { sv ->
            hamVectors.map { hv ->
                SimilarityFuns.simComponentL1DistWeights(hv, sv, weights)
            }.min()!!
        }.average()!!

        val vBad2 = hamVectors.map { sv ->
            spamVectors.map { hv ->
                SimilarityFuns.simComponentL1DistWeights(hv, sv, weights)
            }.min()!!
        }.average()!!


        return vBad * vBad2
    }

    fun getLinearDiscriminant(weights: List<Double>): Double {
//        val componentWeights = weights.take(weights.size - 1)
        val componentWeights = weights.take(weights.size)
//        val offsetWeight = weights.last()
        val offsetWeight = 0.0

        var tp = 0.0
        var tn = 0.0
        var fn = 0.0
        var fp = 0.0

//        val correctSpam = spamVectors.sumByDouble { sv ->
//            if ((SimilarityFuns.dotProduct(sv, componentWeights) + offsetWeight) >= 0.0) 1.0 else 0.0
//        }

        val correctSpam = spamVectors.forEach { sv ->
            if ((SimilarityFuns.dotProduct(sv, componentWeights)  + 0 * offsetWeight) >= 0.5) { tp += 1.0; } else { fn += 1.0; }
        }

//        val correctHam = hamVectors.sumByDouble { sv ->
//            if ((SimilarityFuns.dotProduct(sv, componentWeights) + offsetWeight) < 0.0) 1.0 else 0.0
//        }

        val correctHam = hamVectors.forEach { sv ->
            if ((SimilarityFuns.dotProduct(sv, componentWeights)  + 0 * offsetWeight) < 0.5) { tn += 1.0 }  else { fp += 1.0; }
        }

        val precision = tp.toDouble() / (tp + fp)
        val recall = tp.toDouble() / (tp + fn)
        val f1 = (2 * (precision * recall) / (precision + recall)).run { if(isNaN()) 0.0 else this }
        val precision2 = tn.toDouble() / (tn + fn)
        val recall2 = tn.toDouble() / (tn + fp)
        val f2 = (2 * (precision2 * recall2) / (precision2 + recall2)).run { if(isNaN()) 0.0 else this }

//        return (correctHam / hamVectors.size) *   (correctSpam / spamVectors.size)
//        return correctHam + correctSpam
        return (f1 + f2) / 2.0
    }

    fun getKNN(weights: List<Double>): Double {
        return trainingVectorComponent.getF1(knnLabeler(weights, trainingVectorComponent.holdout))
    }


    fun doTrain() {
//        val descender = StochasticDescent(basisVectors.size, this::getAverageDist, onlyPos = false)
        val descender = StochasticDescent(basisVectors.size, this::getKNN, onlyPos = true)
        descender.search()
//        descender.search { weights ->
//            val score = trainingVectorComponent.getF1(knnLabeler(weights, trainingVectorComponent.holdout))
//            println("WEEE: $score")
//        }
    }

}

fun weightLabeler(weights: List<Double>) =  { e: EmailSparseVector ->
    val score = SimilarityFuns.dotProduct(e, weights)
    if (score > 0.5) "spam" else "ham"
}

fun knnLabeler(weights: List<Double>, vectors: List<EmailSparseVector>) =  { e: EmailSparseVector ->
    vectors.map { it to SimilarityFuns.simComponentL1DistWeights(e, it, weights) }
        .sortedBy { it.second }
        .take(3)
        .map { it.first.label }
        .groupingBy { it }
        .eachCount()
        .maxBy { it.value }!!.key
}
