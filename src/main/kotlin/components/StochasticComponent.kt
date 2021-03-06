package components

import com.google.common.util.concurrent.AtomicDouble
import containers.EmailEmbeddedVector
import containers.EmailSparseVector
import kernels.GaussianKernel
import kernels.SimilarityFuns
import learning.stochastic.SimpleDescent
import org.apache.commons.math3.distribution.NormalDistribution
import org.apache.commons.math3.special.Erf
import utils.*
import java.io.File
import java.util.*
import kotlin.math.absoluteValue
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.system.measureTimeMillis

class StochasticComponent(val nBasis: Int,
                          trainingVectors: List<EmailSparseVector>,
                          var holdout: List<EmailSparseVector>,
                          val nPartitions: Int = 10) {

    private val allSpams = trainingVectors.filter { it.label == "spam" }
    var spamVectors = allSpams



    private val ones = (0 until nBasis).map { 1.0 }
//    val debugOut = File("results.tsv").bufferedWriter()




    private val allHams = trainingVectors.filter { it.label == "ham" }
    var hamVectors = allHams
//    val ghettoTree = GhettoKDTree(trainingVectorComponent)

    var memoizedSpamDist = NormalDistribution(0.1, 2.0)
    private var memoizedSpamDists = createNormalDists(ones, spamVectors)
    private val memoizedSpamDists2 = createNormalDistPartitions(ones, spamVectors, 10)

    var memoizedHamDist = NormalDistribution(0.1, 2.0)
    private var memoizedHamDists = createNormalDists(ones, hamVectors)
    private val memoizedHamDists2 = createNormalDistPartitions(ones, hamVectors, 10)

    val hamCovariance = generateCovarianceMatrix(hamVectors)
    val spamCovariance = generateCovarianceMatrix(spamVectors)

    fun getAverageDist(weights: List<Double>): Double {
        val w2 = weights
        val spamDist = createNormalDist(w2, spamVectors)

        val variance = spamDist.standardDeviation.pow(2.0)
        val nComponents = spamVectors.first().components.size
        val theoreticalVariance = spamVectors.first().components.keys.sumByDouble { component ->
            val weight = weights[component.toInt()]
            (0 until nComponents).sumByDouble { c2 ->
                spamCovariance[setOf(component, c2.toString())]!! * weight * weights[c2.toInt()]
            }

        }
        println("${variance / spamVectors.size} : $theoreticalVariance")

        val hamDist = createNormalDist(w2, hamVectors)
        val transformed = (spamVectors + hamVectors).map { SimilarityFuns.dotProduct(it.components, w2) }
        return getDistance4(spamDist, hamDist, transformed)
    }

    fun getAverageDist2(weights: List<Double>): Double {
        val w2 = weights
        val spamDist = createNormalDist(w2, spamVectors)
        val hamDist = createNormalDist(w2, hamVectors)
//        val transformed = (spamVectors + hamVectors).map { SimilarityFuns.dotProduct(it.components, w2) }
//        val spamPoints = (spamVectors).map { SimilarityFuns.dotProduct(it.components, w2) }
//            .joinToString(" ")
//        val hamPoints = (hamVectors).map { SimilarityFuns.dotProduct(it.components, w2) }
//            .joinToString(" ")
        val transformed = (holdout).map { SimilarityFuns.dotProduct(it.components, w2) }

//        debugOut.write("${hamDist.mean}\t${hamDist.standardDeviation}\t${spamDist.mean}\t${spamDist.standardDeviation}\t$hamPoints\t$spamPoints\n")

        return getDistance4(spamDist, hamDist, transformed)
    }

    fun foldDists(dists: Map<String, NormalDistribution>, weights: List<Double>): NormalDistribution {
        var s1 = 0.0
        var s2 = 0.0

        dists.forEach { (component, dist) ->
            val weight = weights[component.toInt()]
            s1 += weight * (dist.mean.pow(2.0) + dist.standardDeviation.pow(2.0))
            s2 += weight * dist.mean
        }

        val s3 = s2.pow(2.0)

        val finalSd = Math.sqrt((s1 - s3).absoluteValue)
        return NormalDistribution(s2, finalSd)
    }

    fun generateCovarianceMatrix(vectors: List<EmailSparseVector>): HashMap<Set<String>, Double> {
//        val sds = HashMap<String, Double>()
        val results = vectors.first().components
            .map { component ->
                val values = vectors.map { it.components[component.key]!! }
                val mean = values.average()
                val geoMean = values.reduce { acc, cur -> acc * (if (cur == 0.0) 1.0 else cur) }
                    .pow(1.0/(values.count { it != 0.0 }))
//                println("$geoMean : $mean")
//                val variance = values.sumByDouble { (mean - it).pow(2.0) } / values.size
//                sds[component.key] = variance.pow(0.5)
                Triple(component.key, mean, values)
            }


        val covarianceMatrix = HashMap<Set<String>, Double>()

        results.forEach { (r1, mean1, values1) ->
            results.onEach { (r2, mean2, values2) ->
                val covariance =  values1.zip(values2).sumByDouble { (v1, v2) -> (v1 - mean1) * (v2 - mean2) } / values1.size
//                covarianceMatrix[setOf(r1, r2)] = covariance / (sds[r1]!! * sds[r2]!!)
                covarianceMatrix[setOf(r1, r2)] = covariance
            }
        }
        return covarianceMatrix

//        covarianceMatrix.entries
//            .sortedByDescending { it.value.absoluteValue }
//            .forEach { println("${it.key} : ${it.value}") }

    }


    fun getComponentsDists(weights: List<Double>): Double {
        val w2 = weights
//        val spamDist = createNormalDists(w2, spamVectors)
//        val hamDist = createNormalDists(w2, hamVectors)
        val transformed = (spamVectors + hamVectors).map { SimilarityFuns.dotProduct(it.components, weights) }


        val spamDist = foldDists(memoizedSpamDists, weights)
        val hamDist  = foldDists(memoizedHamDists, weights)

        val lf1 = transformed.map { spamDist.getPerturb(it) }.map { if (it <= 0.0) 0.0001 else it  }.normalize()
        val lf2 = transformed.map { hamDist.getPerturb(it) }.map{if (it <= 0.0) 0.0001 else it}.normalize()


        val d1 = symKldDist3(lf1, lf2)
        val uniform = transformed.map { 1.0 }.normalize()
        val d3 = -symKldDist3(lf1, uniform)
        val d4 = -symKldDist3(lf2, uniform)
        return d1 + (d3 + d4)
    }

    fun rbfF1Score(weights: List<Double>): Double {
        val result =  getF1(myLabeler2(weights))
        return result
    }

    fun linearDiscrim(weights: List<Double>): Double {
        val result =  getF1(myDiscrimLabeler(weights))
        return result
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
        val d1 = lf1.zip(lf2).sumByDouble { (v1, v2) ->
            val div1 = v1 * Math.log(v1 / v2)
            val div2 = v2 * Math.log(v2 / v1)
            (div1 + div2) / 2.0
        }

        val uniform = points.map { 1.0  }.normalize()
        val d3 = -(lf1.zip(uniform).sumByDouble {  (it.first * Math.log(it.first / it.second)) })
        val d4 = -(lf2.zip(uniform).sumByDouble {  (it.first * Math.log(it.first / it.second)) })

        return d1 + (d3 + d4)
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

    fun kld(d1: List<Double>, d2: List<Double>): Double = d1.zip(d2).sumByDouble { (v1, v2) ->
        v1 * Math.log(v1 / v2)
    }

    fun symKldDist(d1: List<Double>, d2: List<Double>): Double = d1.zip(d2).sumByDouble { (v1, v2) ->
        val div1 = (v1 * Math.log(v1) + v2 * Math.log(v2))
        val div2 = (v1 + v2) * Math.log(v1 + v2)
        div1 - div2
    }

    fun symKldDist2(d1: List<Double>, d2: List<Double>): Double {
        val midpoint = d1.zip(d2).map { it.first * 0.5 + it.second * 0.5 }
        return kld(d1, midpoint) + kld(d2, midpoint)
    }

    fun symKldDist3(d1: List<Double>, d2: List<Double>): Double {
        val midpoint = d1.zip(d2).map { it.first * 0.5 + it.second * 0.5 }
        return kld(d1, midpoint) * 0.5 + kld(d2, midpoint) * 0.5
    }

    fun getDistance4(dist1: NormalDistribution, dist2: NormalDistribution, points: List<Double>): Double {
        val lf1 = points.map { (dist1.getPerturb(it).run { if (this == 0.0) 0.00000001 else this })}.normalize()
        val lf2 = (points).map { (dist2.getPerturb(it).run { if (this == 0.0) 0.00000001 else this})}.normalize()
        val d1 = symKldDist3(lf1, lf2)
        val uniform = points.map { 1.0  }.normalize()
        val d3 = -symKldDist3(lf1, uniform)
        val d4 = -symKldDist3(lf2, uniform)

        return d1 + (d3 + d4)
    }








    fun NormalDistribution.getInvDist(point: Double): Double {
        val dist = (point - mean).absoluteValue
        val p1 = probability(mean - dist, mean + dist)
        return 1.0 - p1
    }

    fun NormalDistribution.getPerturb(point: Double): Double {
        val dist = (point - mean).absoluteValue
//        return 1 / (1 + Math.exp((dist.pow(2.0) / (2 * standardDeviation.pow(2.0)))))
//        return 1 / (Math.exp(dist - standardDeviation))
       return Erf.erfc(dist / (Math.sqrt(2.0) * standardDeviation))
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

        holdout.forEachParallelQ { v ->
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
        val descender = SimpleDescent(nBasis, this::getAverageDist, onlyPos = false, useDist = false, winnow = winnow, endFun = { weights ->
//            val descender = SimpleDescent(nBasis, this::getComponentsDists, onlyPos = false, useDist = false, winnow = winnow, endFun = {
            counter += 1
            spamVectors = allSpams
            hamVectors = allHams

//            getAverageDist2(weights)

        })

//        generateCovarianceMatrix(spamVectors)

        return descender.search(nIterations) { weights ->
            memoizedHamDist = createNormalDist(weights, allHams)
            memoizedSpamDist = createNormalDist(weights, allSpams)

//            memoizedHamDist =  foldDists(memoizedHamDists, weights)
//            memoizedSpamDist = foldDists(memoizedSpamDists, weights)

//            memoizedSpamDists = createNormalDists(weights, allHams)
//            memoizedHamDists = createNormalDists(weights, allSpams)

            println("F1: ${getKNN(weights)}") }
    }

    fun doTrain2(winnow: Boolean = true, nIterations: Int = 600): List<Double> {
        val descender = SimpleDescent(nBasis, this::rbfF1Score, onlyPos = false, useDist = false, winnow = winnow)

        return descender.search(iterations = nIterations)
    }

    fun doTrain3(winnow: Boolean = true, nIterations: Int = 600): List<Double> {
        val descender = SimpleDescent(nBasis + 1, this::linearDiscrim, onlyPos = false, useDist = false, winnow = winnow)

        return descender.search(iterations = nIterations)
    }




    fun createNormalDist(weights: List<Double>, vectors: List<EmailSparseVector>): NormalDistribution {
        val scores = vectors.map { SimilarityFuns.dotProduct(it.components, weights) }
        val average = scores.average()
//        val geoAverage = 10.0.pow(scores.filter { it != 0.0 }.map { Math.log(it.absoluteValue) }.average())
//        val average = (10.0.pow(scores.sumByDouble { Math.log10(it.absoluteValue) } / scores.size))

        val variance = scores.map { (average - it).pow(2.0) }.sum()
//        println(variance / average)
        return NormalDistribution(average, variance.pow(0.5).run { if (this <= 0.0) 0.001 else this })
    }

    fun createNormalDists(weights: List<Double>, vectors: List<EmailSparseVector>): HashMap<String, NormalDistribution> {
        val normDists = HashMap<String, NormalDistribution>()

        vectors.first().components.keys.forEach { component ->
            val weight = weights[component.toInt()]
            val scores = vectors.map { vector -> vector.components[component]!! * weight  }
//            val mean = scores.average() * weight
            val mean = scores.average()
//            val variance = scores.map { (mean - it).pow(2.0) }.sum() / (scores.size - 1.0)
            val variance = scores.map { (mean - it).pow(2.0) }.sum()
            normDists[component] = NormalDistribution(mean, variance.pow(0.5).run { if (this <= 0.0) 0.0000001 else this })

        }
        return normDists
    }

    fun createNormalDistPartitions(weights: List<Double>, vectors: List<EmailSparseVector>, nPartitions: Int = 5): HashMap<String, List<NormalDistribution>> {
        val normDists = HashMap<String, List<NormalDistribution>>()

        vectors.first().components.keys.forEach { component ->
            val scores = vectors.map { vector -> vector.components[component]!!  }
            val weight = weights[component.toInt()]

            val dists = scores.chunked(scores.size / nPartitions)
                .map { partition ->
                    val mean = partition.average() * weight
                    val variance = partition.map { (mean - it).pow(2.0) }.sum()
                    NormalDistribution(mean, variance.pow(0.5).run { if (this <= 0.0) 0.001 else this })
                }

            normDists[component] = dists
        }
        return normDists
    }


    fun myLabeler(weights: List<Double>) = { e: EmailSparseVector ->
        val w2 = weights
        val point = SimilarityFuns.dotProduct(e.components, w2)
//        if (memoizedSpamDist.getPerturb(point) > memoizedHamDist.getPerturb(point)) "spam" else "ham"
        if ((memoizedSpamDist.mean - point).absoluteValue < (memoizedHamDist.mean - point).absoluteValue) "spam" else "ham"


//        ghettoTree.retrieveCandidates(e, weights, 5)
//            .map { it.label }
//            .groupingBy { it }
//            .eachCount()
//            .maxBy { it.value }!!.key
    }


    fun myLabeler2(weights: List<Double>) = { e: EmailSparseVector ->
        var hamScore = 0.0
        var spamScore = 0.0
//            e.components.forEach { (k, v) ->
//                val weight = weights[k.toInt()]
//                hamScore += weight * Math.log(memoizedHamDists[k]!!.getInvDist(v)).defaultWhenNotFinite(0.0)
//                spamScore += weight * Math.log(memoizedSpamDists[k]!!.getInvDist(v)).defaultWhenNotFinite(0.0)
//            }

        e.components.flatMap { (k, v) ->
            val weight = weights[k.toInt()]
            val hams = memoizedHamDists2[k]!!.map { "ham" to weight * it.getInvDist(v) }
            val spams = memoizedSpamDists2[k]!!.map { "spam" to weight * it.getInvDist(v) }
            hams + spams
        }.maxBy { it.second }!!
            .first
//        if (hamScore > spamScore) "ham" else "spam"
    }

    fun myDiscrimLabeler(weights: List<Double>) = { e: EmailSparseVector ->
        val point = SimilarityFuns.dotProduct(e.components, weights)
        if ((point - weights.last()) > 0.0) "spam" else "ham"
    }



}

fun weightLabeler(weights: List<Double>) =  { e: EmailSparseVector ->
    val score = SimilarityFuns.dotProduct(e.components, weights)
    if (score > 0.5) "spam" else "ham"
}

fun knnLabeler(weights: List<Double>, vectors: List<EmailSparseVector>) =  { e: EmailSparseVector ->
    vectors.map { it to SimilarityFuns.simComponentL1DistWeights(e.components, it.components, weights) }
        .sortedBy { it.second }
        .take(3)
        .map { it.first.label }
        .groupingBy { it }
        .eachCount()
        .maxBy { it.value }!!.key
}


fun main(args: Array<String>) {
    val point = 2.2
    val mean = 2.0
    val sd = 0.25
    val normal = NormalDistribution(mean, sd)


    val dist = (point - mean).absoluteValue
    val p1 = normal.probability(mean - dist, mean + dist)
    val result = 1.0 - p1
    val x1 = Erf.erfc(dist / (Math.sqrt(2.0) * sd ))
    val x2 = -2.0/((Math.sqrt(Math.PI))) * Math.exp(-(x1.pow(2.0)))

    println(result)
    println(x2 / x1)




}