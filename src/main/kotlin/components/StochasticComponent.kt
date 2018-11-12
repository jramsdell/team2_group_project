package components

import containers.EmailSparseVector
import kernels.SimilarityFuns
import learning.GhettoKDTree
import learning.stochastic.StochasticDescent
import org.apache.commons.math3.distribution.NormalDistribution
import kotlin.math.absoluteValue
import kotlin.math.pow

class StochasticComponent(val basisVectors: List<EmailSparseVector>,
                          trainingVectors: List<EmailSparseVector>,
                          val trainingVectorComponent: TrainingVectorComponent) {
    val spamVectors = trainingVectors.filter { it.label == "spam" }
    val hamVectors = trainingVectors.filter { it.label == "ham" }
    val ghettoTree = GhettoKDTree(trainingVectorComponent)



    fun getAverageDist(weights: List<Double>): Double {
        val dots = (spamVectors + hamVectors).map { it to SimilarityFuns.dotProduct(it, weights) }.toMap()

//        val vBad = spamVectors.map { sv ->
////            val svBase = SimilarityFuns.dotProduct(sv, weights)
//            val svBase = dots[sv]!!
//            hamVectors.map { hv ->
////                Math.log(SimilarityFuns.simComponentL2DistWeights(hv, sv, weights)).run { if(isFinite()) this else -1.0 }
////                if (svBase > SimilarityFuns.dotProduct(hv, weights)) 1.0 else 0.0
//                if (svBase > dots[hv]!!) 1.0 else 0.0
////                svBase - dots[hv]!!
////                if (svBase > dots[hv]!!) 1.0 else 0.0
//            }.sum()
//        }.sum()

        val spamDist = createNormalDist(weights, spamVectors)
        val hamDist = createNormalDist(weights, hamVectors)
        val vBad = spamDist.sample(100)
            .map { 1.0 - hamDist.getInvDist(it) }
            .average()

//        val vGood = hamVectors.map { hv ->
//            val hvBase = dots[hv]!!
//            spamVectors.map { sv ->
//                //                Math.log(SimilarityFuns.simComponentL2DistWeights(hv, sv, weights)).run { if(isFinite()) this else -1.0 }
//                if (hvBase < dots[sv]!!) 1.0 else 0.0
//            }.sum()
//        }.sum()

//        val vBad2 = hamVectors.map { sv ->
//            spamVectors.map { hv ->
//                SimilarityFuns.simComponentL1DistWeights(hv, sv, weights)
//            }.min()!!
//        }.average()!!


        return vBad
    }

    fun NormalDistribution.getInvDist(point: Double): Double {
        val dist = (point - mean).absoluteValue
        val p1 = probability(mean - dist, mean + dist)
//    val p2 = probability(mean - dist, mean)
        return 1.0 - p1
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
//        return trainingVectorComponent.getF1(knnLabeler(weights, trainingVectorComponent.holdout))
        return trainingVectorComponent.getF1(myLabeler(weights, trainingVectorComponent.holdout))
    }


    fun doTrain() {
//        val descender = StochasticDescent(basisVectors.size, this::getAverageDist, onlyPos = false)
        val descender = StochasticDescent(basisVectors.size, this::getAverageDist, onlyPos = false, useDist = false)
        descender.search({weights -> println(getKNN(weights)) })
//        descender.search { weights ->
//            val score = trainingVectorComponent.getF1(knnLabeler(weights, trainingVectorComponent.holdout))
//            println("WEEE: $score")
//        }
    }

    fun createNormalDist(weights: List<Double>, vectors: List<EmailSparseVector>): NormalDistribution {
        val scores = vectors.map { SimilarityFuns.dotProduct(it, weights) }
        val average = scores.average()
        val variance = scores.map { (average - it).pow(2.0) }.sum()
        return NormalDistribution(average, variance.pow(0.5))
    }

    fun myLabeler(weights: List<Double>, vectors: List<EmailSparseVector>) = { e: EmailSparseVector ->
//        val avHam = hamVectors.map { SimilarityFuns.dotProduct(it, weights) }.average()
        val distHam = createNormalDist(weights, hamVectors)
//        val avSpam = spamVectors.map { SimilarityFuns.dotProduct(it, weights) }.average()
        val distSpam = createNormalDist(weights, spamVectors)
        val point = SimilarityFuns.dotProduct(e, weights)


//        if ((point - avHam).absoluteValue < (point - avSpam).absoluteValue) "ham" else "spam"
        if (distSpam.getInvDist(point) > distHam.getInvDist(point)) "spam" else "ham"
//        ghettoTree.retrieveCandidates(e, weights, 5)
//            .map { it.label }
//            .groupingBy { it }
//            .eachCount()
//            .maxBy { it.value }!!.key
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
