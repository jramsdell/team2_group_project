package components

import containers.EmailSparseVector
import kernels.SimilarityFuns
import learning.GhettoKDTree
import learning.stochastic.SimpleDescent
import learning.stochastic.StochasticDescent
import org.apache.commons.math3.distribution.NormalDistribution
import org.apache.commons.math3.random.RandomGenerator
import utils.*
import java.util.*
import kotlin.math.absoluteValue
import kotlin.math.pow

class StochasticComponent(val basisVectors: List<EmailSparseVector>,
                          trainingVectors: List<EmailSparseVector>,
                          val trainingVectorComponent: TrainingVectorComponent,
                          val nPartitions: Int = 10) {
    val perturber = NormalDistribution(org.apache.commons.math3.random.MersenneTwister(123), 0.0, 1.0)
    val spamVectors = trainingVectors.filter { it.label == "spam" }
//        .map {
//            val newComponents = it.components.map { (k,v) -> k to v + (0.01 * v * perturber.sample()) }
//                .toMap()
//            EmailSparseVector(it.label, newComponents, it.id)
//        }

    val hamVectors = trainingVectors.filter { it.label == "ham" }
//        .map {
//            val newComponents = it.components.map { (k,v) -> k to v + (0.01 * v * perturber.sample()) }
//                .toMap()
//            EmailSparseVector(it.label, newComponents, it.id)
//        }
//    val ghettoTree = GhettoKDTree(trainingVectorComponent)
    val partitionMappings = (0 until basisVectors.size)
        .shuffled(Random(1231))
        .windowed(basisVectors.size / nPartitions, basisVectors.size / nPartitions, true)
        .mapIndexed { index, list -> list.map { it to index }  }
        .flatten()
        .toMap()


    fun getSpamHamMeans(w2: List<Double>): Pair<Map<Int, Double>, Map<Int, Double>> {
        val spamMeans = spamVectors.flatMap { SimilarityFuns.dotProducts(it, w2, partitionMappings).toList() }
            .groupBy { it.first }
            .map { it.key to it.value.sumByDouble { it.second } / it.value.size }
            .toMap()

        val hamMeans = hamVectors.flatMap { SimilarityFuns.dotProducts(it, w2, partitionMappings).toList() }
            .groupBy { it.first }
            .map { it.key to it.value.sumByDouble { it.second } / it.value.size }
            .toMap()

        return spamMeans to hamMeans

    }

    fun getAverageDist(weights: List<Double>): Double {
//        val dots = (spamVectors + hamVectors).map { it to SimilarityFuns.dotProduct(it, weights) }.toMap()
        val w2 = weights
//        val (spamMeans, hamMeans) = getSpamHamMeans(w2)

        val spamDist = createNormalDist(w2, spamVectors)
        val hamDist = createNormalDist(w2, hamVectors)
        val transformed = (spamVectors + hamVectors).map { SimilarityFuns.dotProduct(it, w2) }
//        val vectors = hamVectors + spamVectors

//        val result = (0 until basisVectors.size)
//            .map { component ->
//                val c = component.toString()
//                Math.log(getDistance2(spamDist[c]!!, hamDist[c]!!, vectors.map { it.components[c]!! * weights[component] }) )
//            }.sum()

        return getDistance2(spamDist, hamDist, transformed)
//        return result
//        return getDistance(spamDist.mean, hamDist.mean, transformed)

//        val spamDists = createNormalDists(weights, spamVectors)
//        val hamDists = createNormalDists(weights, hamVectors)
//        val transformed = (spamVectors + hamVectors).map { SimilarityFuns.dotProducts(it, weights, partitionMappings) }
//        return (0 until nPartitions).sumByDouble { partition ->
//            val hamMean = hamMeans[partition]!!
//            val spamMean = spamMeans[partition]!!
//            val points = transformed.map { it[partition]!! }
//            getDistance(hamMean, spamMean, points).defaultWhenNotFinite(0.0)
//        }


//        return (spamDist.mean - hamDist.mean).absoluteValue
//        val uniform = 1.0 / lf1.size.toDouble()
//        return -lf1.zip(lf2).sumByDouble { it.first * Math.log(it.first /  uniform)  * it.second * Math.log(it.second / uniform) * (it.first / it.second).absoluteValue}
//        return -lf1.zip(lf2).sumByDouble { it.first * Math.log(it.first /  uniform)  * Math.log((it.second / it.first))}
//        return lf1.zip(lf2).sumByDouble { it.first * Math.log(it.first / it.second)  } +
//                 lf2.zip(lf2).sumByDouble { it.second * Math.log(it.second / it.first)  } +
//                lf2.zip(lf2).sumByDouble { it.first * Math.log(it.second / it.first)  } +
    }


    fun getPartitionDist(weights: List<Double>, partition: Int): Double {
        val w2 = weights
        val (spamMeans, hamMeans) = getSpamHamMeans(w2)
        val transformed = (spamVectors + hamVectors).map { SimilarityFuns.dotProducts(it, weights, partitionMappings) }
        val hamMean = hamMeans[partition]!!
        val spamMean = spamMeans[partition]!!
        val points = transformed.map { it[partition]!! }
        return getDistance(hamMean, spamMean, points).defaultWhenNotFinite(0.0)
    }

    fun getDistance(mean1: Double, mean2: Double, points: List<Double>): Double {
        val lf1 = points.map { (mean1 - it).absoluteValue}.cosine()
        val lf2 = (points).map { (mean2 - it).absoluteValue}.cosine()
        val d1 = -(lf1.zip(lf2).sumByDouble {  (it.first * it.second) }.absoluteValue)
        val uniform = points.map { 1.0 }.cosine()
        val d2 = (lf1.zip(uniform).sumByDouble {  (it.first * it.second) })
        val d3 = (lf2.zip(uniform).sumByDouble {  (it.first * it.second) })

        return d1 + d2 + d3

    }

    fun getDistance2(dist1: NormalDistribution, dist2: NormalDistribution, points: List<Double>): Double {
        val lf1 = points.map { (dist1.getPerturb(it))}.normalize()
        val lf2 = (points).map { (dist2.getPerturb(it))}.normalize()
        val d1 = lf1.zip(lf2).sumByDouble { (v1, v2) -> v1 * Math.log(v1 / v2) }
        val d2 = lf1.zip(lf2).sumByDouble { (v1, v2) -> v2 * Math.log(v2 / v1) }

        val uniform = points.map { 1.0  }.normalize()
        val d3 = -(lf1.zip(uniform).sumByDouble {  (it.first * Math.log(it.first / it.second)) })
        val d4 = -(lf2.zip(uniform).sumByDouble {  (it.first * Math.log(it.first / it.second)) })

//        return Math.exp(d1) * (Math.exp(d2) * Math.exp(d3))
        return (d1 + d2 ) / 2.0 + (d3 + d4)
    }

    fun getDistance3(dist1: NormalDistribution, dist2: NormalDistribution, points: List<Double>): Double {
        val lf1 = points.map { (dist1.getPerturb(it))}.cosine()
        val lf2 = (points).map { (dist2.getPerturb(it))}.cosine()
        val d1 = -(lf1.zip(lf2).sumByDouble {  (it.first * it.second) }.absoluteValue)
        val uniform = points.map { 1.0  }.cosine()
        val d2 = (lf1.zip(uniform).sumByDouble {  (it.first * it.second) })
        val d3 = (lf2.zip(uniform).sumByDouble {  (it.first * it.second) })

//        return Math.exp(d1) * (Math.exp(d2) * Math.exp(d3))
        return d1 +  d2 + d3
    }


    fun NormalDistribution.getInvDist(point: Double): Double {
        val dist = (point - mean).absoluteValue
        val p1 = probability(mean - dist, mean + dist)
        return 1.0 - p1
    }

    fun NormalDistribution.getPerturb(point: Double): Double {
        val dist = (point - mean).absoluteValue
        val p1 = probability(mean - dist, mean + dist)
        return 1.0 - p1
    }



    fun getKNN(weights: List<Double>): Double {
//        return trainingVectorComponent.getF1(knnLabeler(weights, trainingVectorComponent.holdout))
        return trainingVectorComponent.getF1(myLabeler(weights, trainingVectorComponent.holdout))
    }


    fun doTrain(): List<Double> {
        val descender = SimpleDescent(basisVectors.size, this::getAverageDist, onlyPos = false, useDist = false)
        return descender.search({ weights -> println("F1: ${getKNN(weights)}") })
    }

    fun doPartitionTrain(): List<Double> {

        val optimalWeights = ArrayList<List<Double>>()

        (0 until nPartitions).forEach { partition ->
            println("\n\n====PARTITION: $partition =====\n\n")
            val descender = StochasticDescent(basisVectors.size, { weights: List<Double> -> getPartitionDist(weights, partition)}, onlyPos = false, useDist = false)
            val f1Fun = { weights: List<Double> -> trainingVectorComponent.getF1(myPartitionLabeler(weights, trainingVectorComponent.holdout, partition))}
            optimalWeights += descender.search({ weights -> println("F1: ${f1Fun(weights)}") })
        }
//
        println("Final Result\n\n")

        val result = trainingVectorComponent.getF1 { email ->
            var hamScore = 0.0
            var spamScore = 0.0


//            optimalWeights.forEachIndexed { index, weights ->
//                val label = myPartitionLabeler2(weights, emptyList(), index)(email)
//                if (label == "ham") hamScore += 1.0 else spamScore += 1.0
//            }

            val label = optimalWeights.mapIndexed { index, weights ->
                myPartitionLabeler2(weights, emptyList(), index)(email) }
                .forEach {
                    spamScore += it.first
                    hamScore += it.second
                }


            if (hamScore < spamScore) "ham" else "spam"
        }
        println(result)

        return emptyList()
    }

    fun createNormalDist(weights: List<Double>, vectors: List<EmailSparseVector>): NormalDistribution {
        val scores = vectors.map { SimilarityFuns.dotProduct(it, weights) }
        val average = scores.average()
        val variance = scores.map { (average - it).pow(2.0) }.sum()
        return NormalDistribution(average, variance.pow(0.5))
    }

    fun createNormalDists(weights: List<Double>, vectors: List<EmailSparseVector>): HashMap<String, NormalDistribution> {
        val normDists = HashMap<String, NormalDistribution>()

        vectors.first().components.keys.forEach { component ->
            val weight = weights[component.toInt()]
            val scores = vectors.map { vector -> vector.components[component]!!  }
            val mean = scores.average() * weight
            val variance = scores.map { (mean - it).pow(2.0) }.sum()
            normDists[component] = NormalDistribution(mean, variance.pow(0.5))

        }
        return normDists
    }

    fun myPartitionLabeler(weights: List<Double>, vectors: List<EmailSparseVector>, partition: Int) = { e: EmailSparseVector ->
        val w2 = weights
        val points = SimilarityFuns.dotProducts(e, w2, partitionMappings)

        val (spamMeans, hamMeans) = getSpamHamMeans(w2)
        val point = points[partition]!!
        val sDist = (spamMeans[partition]!! - point).absoluteValue
        val hDist = (hamMeans[partition]!! - point).absoluteValue
        if (sDist < hDist) "spam" else "ham"
    }

    fun myPartitionLabeler2(weights: List<Double>, vectors: List<EmailSparseVector>, partition: Int) = { e: EmailSparseVector ->
        val w2 = weights
        val points = SimilarityFuns.dotProducts(e, w2, partitionMappings)

        val (spamMeans, hamMeans) = getSpamHamMeans(w2)
        val point = points[partition]!!
        val sDist = (spamMeans[partition]!! - point).absoluteValue
        val hDist = (hamMeans[partition]!! - point).absoluteValue
//        if (sDist < hDist) "spam" to sDist / hDist else "ham" to (hDist / sDist)
        Math.log(sDist / hDist).defaultWhenNotFinite(0.0) to Math.log(hDist / sDist).defaultWhenNotFinite(0.0)
    }

    fun myLabeler(weights: List<Double>, vectors: List<EmailSparseVector>) = { e: EmailSparseVector ->
        val w2 = weights
//        val avHam = hamVectors.map { SimilarityFuns.dotProduct(it, weights) }.average()
        val distHam = createNormalDist(w2, hamVectors)
//        val avSpam = spamVectors.map { SimilarityFuns.dotProduct(it, weights) }.average()
        val distSpam = createNormalDist(w2, spamVectors)
        val point = SimilarityFuns.dotProduct(e, w2)
//        val points = SimilarityFuns.dotProducts(e, w2, partitionMappings)

//        val (spamMeans, hamMeans) = getSpamHamMeans(w2)

//        val hamDists = createNormalDists(weights, hamVectors)
//        val spamDists  = createNormalDists(weights, spamVectors)


        var hamScore = 1.0
        var spamScore = 1.0
//
//        hamDists.keys.forEach { key ->
//            val p = e.components[key]!!
//            val weight = weights[key.toInt()]
//            val hDist = hamDists[key]!!.getPerturb(p * weight)
//            val sDist = spamDists[key]!!.getPerturb(p * weight)
//            hamScore += hDist - sDist
//            spamScore += sDist - hDist
//        }
//
//        if (spamScore > hamScore) "spam" else "ham"

//        if ((point - distHam.mean).absoluteValue < (point - distSpam.mean).absoluteValue) "ham" else "spam"
        if (distSpam.getPerturb(point) > distHam.getPerturb(point)) "spam" else "ham"

//        (0 until nPartitions).forEach { partition ->
//            val p = points[partition]!!
//            val sDist = (spamMeans[partition]!! - p).absoluteValue
//            val hDist = (hamMeans[partition]!! - p).absoluteValue
//            val vDiff = (sDist - hDist).absoluteValue
////            hamScore += hDist * Math.log(hDist / sDist).defaultWhenNotFinite(0.0)
////            spamScore += sDist * Math.log(sDist / hDist).defaultWhenNotFinite(0.0)
//            hamScore += hDist / sDist
//            spamScore += sDist / hDist
////            if (hDist < sDist) hamScore += vDiff else spamScore += vDiff
////            hamScore *= hDist / vDiff
////            spamScore *= sDist / vDiff
//        }

//        if (hamScore < spamScore) "ham" else "spam"

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
