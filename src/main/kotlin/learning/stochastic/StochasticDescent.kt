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


class StochasticDescent(val nFeatures: Int, val scoreFun: (List<Double>) -> Double, val onlyPos: Boolean = false, val useDist: Boolean = true) {
    val covers = (0 until nFeatures).map { NormalCover().apply { createBalls() } }

    val nCandidates = 10
    val bestMap = (0 until nCandidates).map { -999999.9 }.toArrayList()
    var iterations = 0
    var mapCounter = 0


    fun getBalls() = covers.pmap { it.draw() }
    fun newGeneration(nChildren: Int) = covers.forEachParallelQ { cover -> cover.newGeneration(children = nChildren) }
    var highest = -99999.0
    var bestWeights: List<Double> = emptyList()

    fun NormalDistribution.getInvDist(point: Double): Double {
        val dist = (point - mean).absoluteValue
        val p1 = probability(mean - dist, mean + dist)
//    val p2 = probability(mean - dist, mean)
        return 1.0 - p1
    }

    fun runStep() {
        (0 until 60).forEach {
//            if (it % 10 == 0)
//                println(it)
            val balls=  getBalls()
//            val weights = balls.mapIndexed { index, ball ->
//                ball.getParam().run { if (onlyPos && this < 0.0) 0.0 else this }
//            }.normalize()

            val weights = balls.mapIndexed { index, ball ->
                ball.getParam().run { if (onlyPos && this < 0.0) 0.0 else this }
            }.run { if (useDist) normalize() else this }
            val map = scoreFun(weights)

            val bestAv = bestMap.average()
            val margin = bestMap.map { (it - bestAv).let { it.pow(2.0) * it.sign }}.average()!!
            val mapMargin = (map - bestAv).let { it.pow(2.0) * it.sign }

            var chanceOfGenerating = 0.0

            if (mapCounter >= 15 + nCandidates) {
                val std = Math.sqrt(bestMap.map { (it - bestAv).let { it.pow(2.0) * it.sign  } }.sum().absoluteValue / 5.0)
                if (std > 0.0) {
                    val myDist2 = NormalDistribution(margin, std)
                    chanceOfGenerating = myDist2.getInvDist(map)
                }
            }

            if (mapMargin > margin)
                chanceOfGenerating = 1.0 - chanceOfGenerating
            else chanceOfGenerating = 0.0

            if (mapCounter < nCandidates || ThreadLocalRandom.current().nextDouble() <= chanceOfGenerating   ) {
                if (mapCounter >= nCandidates) {
                    var rewardAmount = 4.0
                    balls.zip(weights).forEach { (ball, param) -> ball.reward(rewardAmount, rewardParam = param) }
                }

                val lowestIndex = bestMap.withIndex().minBy { it.value }!!.index
                bestMap[lowestIndex] = map

                if (highest < map) {
                    println("$map : $weights")
                    highest = map
                    bestWeights = weights
                    println()
                }
                mapCounter += 1
            } else {
                balls.forEach { it.penalize(0.1) }
            }
        }
    }



    fun search(weightUser: ((List<Double>) -> Unit)? = null): List<Double> {
        var curHighest = highest
        var badCounter = 0

        (0 until 2000).forEach {
            runStep()
            println("Highest: $highest")

            if (highest == curHighest) {
                badCounter += 1
                newGeneration(80)
            } else {
                weightUser?.invoke(bestWeights)
            }
            curHighest = highest
        }

        return bestWeights
    }
}


