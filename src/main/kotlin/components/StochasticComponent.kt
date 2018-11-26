package components

import com.google.common.util.concurrent.AtomicDouble
import containers.EmailEmbeddedVector
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
import kotlin.math.sqrt

class StochasticComponent(val nBasis: Int,
                          trainingVectors: List<EmailSparseVector>,
                          var holdout: List<EmailSparseVector>,
                          val nPartitions: Int = 10) {
    val perturber = NormalDistribution(org.apache.commons.math3.random.MersenneTwister(123), 0.0, 1.0)
    val chunkSize = 200

    var memoizedSpamDist = NormalDistribution(0.1, 2.0)
    var memoizedHamDist = NormalDistribution(0.1, 2.0)

    val spamPartitions = trainingVectors.filter { it.label == "spam" }
        .chunked(chunkSize)

//    val spamVectors = trainingVectors.filter { it.label == "spam" }
    val allSpams = trainingVectors.filter { it.label == "spam" }
//    var spamVectors = spamPartitions.first()
    var spamVectors = allSpams
//    val spamMatrix = spamVectors.flatMap { it.components.map { it.key.toInt() to it.value } }
//        .groupBy { it.first }
//        .map { it.key to it.value.map { it.second } }
//        .sortedBy { it.first }
//        .map { it.second }


//        .map {
//            val newComponents = it.components.map { (k,v) -> k to v + (0.01 * v * perturber.sample()) }
//                .toMap()
//            EmailSparseVector(it.label, newComponents, it.id)
//        }
val hamPartitions = trainingVectors.filter { it.label == "ham" }
    .chunked(chunkSize)

//    val hamVectors = trainingVectors.filter { it.label == "ham" }
//    var hamVectors = hamPartitions.first()
    val allHams = trainingVectors.filter { it.label == "ham" }
    var hamVectors = allHams
//    val ghettoTree = GhettoKDTree(trainingVectorComponent)

//    val hamMatrix = hamVectors.flatMap { it.components.map { it.key.toInt() to it.value } }
//        .groupBy { it.first }
//        .map { it.key to it.value.map { it.second } }
//        .sortedBy { it.first }
//        .map { it.second }

    fun getAverageDist(weights: List<Double>): Double {
        val w2 = weights
        val spamDist = createNormalDist(w2, spamVectors)
        val hamDist = createNormalDist(w2, hamVectors)
        val transformed = (spamVectors + hamVectors).map { SimilarityFuns.dotProduct(it, w2) }
        return getDistance3(spamDist, hamDist, transformed)
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

        return (d1 + d2 ) / 2.0 + (d3 + d4)
    }

    fun getDistance3(dist1: NormalDistribution, dist2: NormalDistribution, points: List<Double>): Double {
        val lf1 = points.map { (dist1.getPerturb(it))}.cosine()
        val lf2 = (points).map { (dist2.getPerturb(it))}.cosine()
        val d1 = -(lf1.zip(lf2).sumByDouble {  (it.first * it.second) }.absoluteValue)
        val uniform = points.map { 1.0  }.cosine()
        val d2 = (lf1.zip(uniform).sumByDouble {  (it.first * it.second) })
        val d3 = (lf2.zip(uniform).sumByDouble {  (it.first * it.second) })

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
//        return trainingVectorComponent.getF1(myLabeler(weights, trainingVectorComponent.holdout))
        return getF1(myLabeler(weights))
    }

    fun getF1(caller: (EmailSparseVector) -> String): Double {
        var tp = AtomicDouble(0.0)
        var tn = AtomicDouble(0.0)
        var fn = AtomicDouble(0.0)
        var fp = AtomicDouble(0.0)

        holdout.forEachParallel { v ->
            val called = caller(v)
            if (v.label == "spam" && called == "spam") { tp.addAndGet(1.0) }
            else if (v.label == "spam" && called == "ham") { fn.addAndGet(1.0) }
            else if (v.label == "ham" && called == "ham") { tn.addAndGet(1.0) }
            else { fp.addAndGet(1.0) }
        }

        val precision = tp.toDouble() / (tp.get() + fp.get())
        val recall = tp.toDouble() / (tp.get() + fn.get())
        val f1 = (2 * (precision * recall) / (precision + recall)).run { if(isNaN()) 0.0 else this }
        val precision2 = tn.toDouble() / (tn.get() + fn.get())
        val recall2 = tn.toDouble() / (tn.get() + fp.get())
        val f2 = (2 * (precision2 * recall2) / (precision2 + recall2)).run { if(isNaN()) 0.0 else this }

        return (f1 + f2) / 2.0
    }


    var counter = 0


    fun doTrain(winnow: Boolean = true, nIterations: Int = 600): List<Double> {
        val descender = SimpleDescent(nBasis, this::getAverageDist, onlyPos = false, useDist = false, winnow = winnow, endFun = {
            counter += 1
            spamVectors = allSpams
            hamVectors = allHams

        })
        return descender.search(nIterations) { weights ->
            memoizedHamDist = createNormalDist(weights, allHams)
            memoizedSpamDist = createNormalDist(weights, allSpams)

            println("F1: ${getKNN(weights)}") }
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


    fun myLabeler(weights: List<Double>) = { e: EmailSparseVector ->
        val w2 = weights
        val point = SimilarityFuns.dotProduct(e, w2)
        if (memoizedSpamDist.getPerturb(point) > memoizedHamDist.getPerturb(point)) "spam" else "ham"


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
