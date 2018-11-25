package learning.stochastic

import org.apache.commons.math3.distribution.NormalDistribution
import utils.toArrayList
import utils.forEachParallelQ

//import utils.
import utils.*
import java.util.concurrent.ThreadLocalRandom
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.absoluteValue
import kotlin.math.pow
import kotlin.math.sign


class SimpleDescent(val nFeatures: Int, val scoreFun: (List<Double>) -> Double, val onlyPos: Boolean = false, val useDist: Boolean = true) {
    var weights = (0 until nFeatures).map { Math.random() }.cosine()
    val converged = AtomicBoolean(false)

    fun getPartialGradient(feature: Int, base: Double): Pair<Double, Double> {
//        val curVal = weights[index]
        val steps = listOf(-0.0001, -0.001, -0.01, 0.0, -0.05, 0.0001, 0.001, 0.01, 0.05)

        return steps.map { step ->
            val nWeights = weights.mapIndexed { fIndex, value -> if(feature == fIndex) value + step else value    }
            step to scoreFun(nWeights) - base }
            .maxBy { it.second }!!
    }


    fun doStep() {
        val base = scoreFun(weights)
//        println("Cur: $base")

        val best = (0 until nFeatures)
            .filter { weights[it] != 0.0 }
            .pmap { feature -> feature to getPartialGradient(feature, base) }
            .maxBy { it.second.second }!!

        if (best.second.first == 0.0) {
            converged.set(true)
            println("Converged!")
            return
        }

//        println("Best at: ${best.first} with step ${best.second.first}")
        weights = weights.mapIndexed { index, value -> if (index == best.first) value + best.second.first else value  }
//            .map { if (it < 0) 0.0 else it }
            .cosine()
    }



    fun search(weightUser: ((List<Double>) -> Unit)? = null): List<Double> {
        (0 until 2000)
            .forEach {
                if (!converged.get()) {
                    doStep()
                    if (it % 20 == 0) {
                        weightUser?.invoke(weights)
                        println(weights)
                    }
                }
            }

        return weights
    }
}


