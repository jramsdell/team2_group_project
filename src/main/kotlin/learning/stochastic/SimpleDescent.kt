package learning.stochastic

import org.apache.commons.math3.distribution.NormalDistribution
import utils.toArrayList
import utils.forEachParallelQ

//import utils.
import utils.*
import java.util.*
import java.util.concurrent.ThreadLocalRandom
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.absoluteValue
import kotlin.math.pow
import kotlin.math.sign

private data class StepResult(
        val feature: Int,
        val step: Double,
        val gradient: Double
)

private fun List<Double>.transform() = this.cosine()


class SimpleDescent(val nFeatures: Int, val scoreFun: (List<Double>) -> Double, val onlyPos: Boolean = false, val useDist: Boolean = true, val endFun: (() -> Unit)? = null, val winnow: Boolean = true) {
//    var weights = (0 until nFeatures).map { -Math.random() }.normalize()
    var weights = (0 until nFeatures).map { 1.0 }.transform()
    val converged = AtomicBoolean(false)
    private val priorities = PriorityQueue<StepResult>(kotlin.Comparator { t1, t2 -> -compareValues(t1.gradient, t2.gradient)  })
    var lastStep = (0 until nFeatures).map { it to 1.0 }.toHashMap()
    var curScore = 0.0

    fun getPartialGradient(feature: Int, base: Double): Pair<Double, Double> {
//        val curVal = weights[index]
//        val steps = listOf(-0.00001, -0.0001, -0.001, -0.01, 0.0, -0.05, -0.25, 0.001, 0.0001, 0.00001, 0.01, 0.05, 0.25)
//        val steps = listOf(-0.0001, -0.001, -0.01, 0.0, -0.05, -0.25, 0.001, 0.0001, 0.01, 0.05, 0.25)
//        val steps = listOf(-0.0001, -0.001, -0.01, 0.0, -0.05, 0.001, 0.0001, 0.01, 0.05)
        val steps = listOf(-0.0001, -0.001, -0.01, 0.0, -0.05, 0.001, 0.0001, 0.01, 0.05)
//        val steps = listOf(-0.0001, -0.001, -0.01, 0.0, -0.05, -0.25, 0.001, 0.0001, 0.01, 0.05, 0.25)
//        val steps = listOf(-0.001, -0.01, 0.0, -0.05, -0.25, 0.001,0.01, 0.05, 0.25)
//            .filter { it.absoluteValue <= lastStep[feature]!!.absoluteValue }

        return steps.pmap { step ->
            val nWeights = weights.mapIndexed { fIndex, value -> if(feature == fIndex) value + step else value    }.transform()
            step to scoreFun(nWeights) - base }
            .maxBy { it.second }!!
    }

    fun initialize() {
        val base = scoreFun(weights)
        (0 until nFeatures)
            .forEach { feature ->
                val result = getPartialGradient(feature, base)
                val stepResult = StepResult(feature, result.first, result.second)
                priorities.add(stepResult)
            }
    }


//    fun doStep() {
//        curScore = scoreFun(weights)
//        val best = (0 until nFeatures)
//            .filter { weights[it] != 0.0 }
//            .pmap { feature -> feature to getPartialGradient(feature, curScore) }
//            .maxBy { it.second.second }!!
//
//        if (best.second.first == 0.0) {
//            converged.set(true)
//            println("Converged!")
//            return
//        }
//
//        lastStep[best.first] = best.second.first
//
//        weights = weights.mapIndexed { index, value -> if (index == best.first) value + best.second.first else value  }
////            .normalize()
//            .cosine()
//    }

    fun doStep2() {
//        val base = scoreFun(weights)
        curScore = scoreFun(weights)
        val next = priorities.poll()
        if (weights[next.feature] == 0.0) {
            return
        }

        val result = getPartialGradient(next.feature, curScore)
//        println(result.second)

        if (result.first == 0.0 && result.second == 0.0) {
//            converged.set(true)
//            println("Converged!")
            return
        }


        weights = weights.mapIndexed { index, value -> if (index == next.feature) value + result.first else value  }
//            .normalize()
            .transform()

        lastStep[next.feature] = result.first
        val newResult = StepResult(feature = next.feature, step = result.first, gradient = result.second)
        priorities.add(newResult)

    }



    fun search(iterations: Int = 600, weightUser: ((List<Double>) -> Unit)? = null): List<Double> {
        initialize()
        (0 until iterations)
            .forEach {
                if (!converged.get() && priorities.isNotEmpty()) {

                    doStep2()
                    endFun?.invoke()
                    if (it > 100) {
                        weights = weights.mapIndexed { index, value ->
                            if ((-0.05 < value && value < 0.05) && lastStep[index]!! < 0.01 && lastStep[index]!! > -0.01) (if (winnow) 0.0 else value) else value }
//                            if ((-0.001 < value && value < 0.001) && lastStep[index]!! < 0.01 && lastStep[index]!! > -0.01) (if (winnow) 0.0 else value) else value }
                    }
                    if (it % 100 == 99 || winnow == false) {
                        weightUser?.invoke(weights)
                        val count = weights.count { it != 0.0 }
                        println("$count : $curScore")
                    }
                }
            }

        return weights
    }
}


