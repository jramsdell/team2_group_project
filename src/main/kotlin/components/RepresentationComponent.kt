package components

import com.google.common.util.concurrent.AtomicDouble
import containers.EmailSparseVector
import learning.stochastic.SimpleDescent
import org.apache.commons.math3.distribution.NormalDistribution
import representations.KernelType
import representations.RepresentationType
import utils.forEachParallelQ
import utils.normalize

class RepresentationComponent(val nBasis: Int,
                              trainingVectors: List<EmailSparseVector>,
                              var holdout: List<EmailSparseVector>) {

    var spamVectors = trainingVectors.filter { it.label == "spam" }
    var hamVectors = trainingVectors.filter { it.label == "ham" }
    val distComponent = DistributionComponent(this)
    val repIndexMap: HashMap<Int, Pair<RepresentationType, KernelType>> = HashMap()
    val repWeightMap: HashMap<Pair<RepresentationType, KernelType>, Double> = HashMap()
    val representationWeightAssigner =
            { key: Pair<RepresentationType, KernelType> -> repWeightMap[key]!! }


//    var uniform = (0 until nBasis).map { 1.0 }.normalize()
//
//
//
//    fun getDistributionDistance(weights: List<Double>): Double {
//        weights.forEachIndexed { index, weight -> repWeightMap[repIndexMap[index]!!] = weight}
//        distComponent.buildDists(representationWeightAssigner)
//
//        val points = distComponent.totalPoints.values.flatten()
//        return _getDistributionDistance(distComponent.totalDistributions["spam"]!!, distComponent.totalDistributions["ham"]!!, points)
//    }
//
//
//    private fun kld(d1: List<Double>, d2: List<Double>): Double = d1.zip(d2)
//        .sumByDouble { (v1, v2) -> v1 * Math.log(v1 / v2) }
//
//    fun jensenShannonDivergence(d1: List<Double>, d2: List<Double>): Double {
//        val midpoint = d1.zip(d2).map { it.first * 0.5 + it.second * 0.5 }
//        return kld(d1, midpoint) * 0.5 + kld(d2, midpoint) * 0.5
//    }
//
//
//    private fun _getDistributionDistance(dist1: Distribution, dist2: Distribution, points: List<Double>): Double {
//        val lf1 = points.map { dist1.sim(it.run { if (this == 0.0) 0.00000001 else this })}.normalize()
//        val lf2 = points.map { dist2.sim(it.run { if (this == 0.0) 0.00000001 else this })}.normalize()
//
//        val d1 = jensenShannonDivergence(lf1, lf2)
//        val d3 = -jensenShannonDivergence(lf1, uniform)
//        val d4 = -jensenShannonDivergence(lf2, uniform)
//        return d1 + (d3 + d4)
//    }
//
//    fun doTrainTotal(nIterations: Int = 600): List<Double> {
//        distComponent.initialize()
//
//        val descender = SimpleDescent(nBasis, this::_getDistributionDistance, onlyPos = false, useDist = false, winnow = true)
//        return descender.search(nIterations) { weights ->
//
//
//            memoizedHamDist = createNormalDist(weights, allHams)
//            memoizedSpamDist = createNormalDist(weights, allSpams)
//
//            println("F1: ${getKNN(weights)}") }
//    }
//
//
//
//    fun f1Labeler(caller: (EmailSparseVector) -> String): Double {
//        val tp = AtomicDouble(0.0)
//        val tn = AtomicDouble(0.0)
//        val fn = AtomicDouble(0.0)
//        val fp = AtomicDouble(0.0)
//
//        holdout.forEachParallelQ { v ->
//            val called = caller(v)
//            if (v.label == "spam" && called == "spam") { tp.addAndGet(1.0) }
//            else if (v.label == "spam" && called == "ham") { fn.addAndGet(1.0) }
//            else if (v.label == "ham" && called == "ham") { tn.addAndGet(1.0) }
//            else { fp.addAndGet(1.0) }
//        }
//
//        val precision = tp.toDouble() / (tp.get() + fp.get())
//        val recall = tp.toDouble() / (tp.get() + fn.get())
//        val f1 = (2 * (precision * recall) / (precision + recall)).run { if(isNaN()) 0.0 else this }
//        val precision2 = tn.toDouble() / (tn.get() + fn.get())
//        val recall2 = tn.toDouble() / (tn.get() + fp.get())
//        val f2 = (2 * (precision2 * recall2) / (precision2 + recall2)).run { if(isNaN()) 0.0 else this }
//
//        return (f1 + f2) / 2.0
//    }

}

