package learning.stochastic

import org.apache.commons.math3.distribution.NormalDistribution
import utils.toArrayList
import utils.forEachParallelQ

//import utils.
import utils.*
import java.util.concurrent.ThreadLocalRandom
import kotlin.math.absoluteValue
import kotlin.math.pow
import kotlin.math.sign


class SimpleDescent(val nFeatures: Int, val scoreFun: (List<Double>) -> Double, val onlyPos: Boolean = false, val useDist: Boolean = true) {
    var weights = (0 until nFeatures).map { 1.0 }

    fun getPartialGradient(feature: Int, base: Double): Pair<Double, Double> {
//        val curVal = weights[index]
        val steps = listOf(-0.001, -0.01, -0.1, -0.5, -2.5, 0.001, 0.01, 0.1, 0.5, 2.5)

        return steps.map { step ->
            val nWeights = weights.mapIndexed { fIndex, value -> if(feature == fIndex) value + step else value    }
            step to scoreFun(nWeights) - base }
            .maxBy { it.second }!!
    }


    fun doStep() {
        val base = scoreFun(weights)
//        println("Cur: $base")

        val best = (0 until nFeatures)
            .pmap { feature -> feature to getPartialGradient(feature, base) }
            .maxBy { it.second.second }!!

        println("Best at: ${best.first} with step ${best.second.first}")
        weights = weights.mapIndexed { index, value -> if (index == best.first) value + best.second.first else value  }
    }



    fun search(weightUser: ((List<Double>) -> Unit)? = null): List<Double> {
        (0 until 2000)
            .forEach {
                doStep()
                if (it % 20 == 0)
                    weightUser?.invoke(weights)
            }

        return weights
    }
}


