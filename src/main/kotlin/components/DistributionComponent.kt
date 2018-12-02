package components

import containers.EmailSparseVector
import kernels.SimilarityFuns
import org.apache.commons.math3.distribution.NormalDistribution
import org.apache.commons.math3.special.Erf
import representations.ComponentKey
import representations.KernelType
import representations.Representation
import representations.RepresentationType
import utils.pmap
import utils.sd
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.absoluteValue
import kotlin.math.pow

data class Distribution(var mean: Double, var sd: Double) {
    fun sim(point: Double): Double {
        val diff = (mean - point).absoluteValue
        return diff / (Math.sqrt(2.0) * sd)
    }

    fun recalculateDist(scores: List<Double>) {
        mean = scores.average()
        sd = scores.sd(mean)
    }
}



class DistributionComponent( val representationComponent: RepresentationComponent ) {
    val distributions: ConcurrentHashMap<String, ConcurrentHashMap<ComponentKey, Distribution>> = ConcurrentHashMap()
    val totalDistributions = ConcurrentHashMap<String, Distribution>()
//    val points = ConcurrentHashMap<ComponentKey, List<Double>>()
//    val totalPoints = ConcurrentHashMap<String, List<Double>>()
    val allVectors = representationComponent.hamVectors + representationComponent.spamVectors

//     fun buildDists(weightLabeler: (Pair<RepresentationType, KernelType>) -> Double) {
//         _buildDist("ham", representationComponent.hamVectors, weightLabeler)
//         _buildDist("spam", representationComponent.spamVectors, weightLabeler)
//     }

//     private fun _buildDist(distName: String, vectors: List<EmailSparseVector>,
//                    weightLabeler: (Pair<RepresentationType, KernelType>) -> Double) {
//        val reps = vectors.first().representations.keys
//         val distsByName = distributions.computeIfAbsent(distName) { ConcurrentHashMap() }
//         val totalDistByName = totalDistributions.computeIfAbsent(distName) { Distribution(0.0, 0.0) }
//
//        val scores = reps.pmap { repKey ->
//            val weight = weightLabeler(repKey)
//            val scores = vectors.map { email ->
//                val rep = email.representations[repKey]!!
//                val score = rep.score * weight
//                rep.cur
//
//
//            }
//            distsByName.computeIfAbsent(repKey) { Distribution(0.0, 0.0) }.recalculateDist(scores)
////            points[repKey] = scores
//            scores.sum()
//        }
//
////         totalPoints[distName] = scores
//         totalDistByName.recalculateDist(scores)
//    }

    fun initialize() {
        val keys = representationComponent.spamVectors.first().representations.keys
        val hamVectors = representationComponent.hamVectors
        val spamVectors  = representationComponent.spamVectors

        keys.forEach { repKey -> updateComponentWeight("ham", hamVectors, repKey, 1.0) }
        keys.forEach { repKey -> updateComponentWeight("spam", spamVectors, repKey, 1.0) }
        rebuildTotalDist("ham", hamVectors)
        rebuildTotalDist("spam", spamVectors)
    }



    private fun rebuildTotalDist(distName: String, vectors: List<EmailSparseVector>) =
        totalDistributions.computeIfAbsent(distName) { Distribution(0.0, 0.0) }.recalculateDist(vectors.map { it.score })


    fun updateComponentWeight(distName: String, vectors: List<EmailSparseVector>,
                           componentKey: ComponentKey, weight: Double,
                                      updateTotalDist: Boolean = false) {
        val distsByName = distributions.computeIfAbsent(distName) { ConcurrentHashMap() }

        val scores = vectors.pmap { email ->
            val rep = email.representations[componentKey]!!
            val newScore = rep.score * weight
            val diff = newScore - rep.curScore
            email.score += diff
            rep.curScore = newScore
            newScore
        }

        distsByName.computeIfAbsent(componentKey) { Distribution(0.0, 0.0) }.recalculateDist(scores)
        if (updateTotalDist) {
            rebuildTotalDist(distName, vectors)
        }
    }


}

fun main(args: Array<String>) {
    val dist = NormalDistribution(0.2, 0.05)
    val dist2 = NormalDistribution(0.1, 0.03)
    val scores = dist.sample(10000).toList()
    val scores2 = dist2.sample(10000).toList()


    val myDist = Distribution(0.0, 0.0)
    val myDist2 = Distribution(0.0, 0.0)
    val myDistCombined = Distribution(0.0, 0.0)

    myDist.recalculateDist(scores)
    myDist2.recalculateDist(scores2)

//    var weights = listOf(0.2, 0.5)

    var baseMean = myDist.mean
    var baseSd = myDist.sd
    var baseMean2 = myDist2.mean
    var baseSd2 = myDist2.sd

    val secondMoment1 = scores.sumByDouble { it.pow(2.0) } / scores.size
    val secondMoment2 = scores2.sumByDouble { it.pow(2.0) } / scores.size


    val recalc = { weights: List<Double> ->
        myDist.recalculateDist(scores.map { it * weights[0] })
        myDist2.recalculateDist(scores2.map { it * weights[1] })
        myDistCombined.recalculateDist(scores.zip(scores2).map { it.first * weights[0] +  it.second * weights[1] })
    }


    val reportResult = { curDist: Distribution, oldMean: Double, oldSd: Double, name: String ->
        with (curDist) {
//            println("$name : $mean ($oldMean),  $sd ($oldSd)")
            println("$name : $sd : ${sd.pow(2.0)}")
        }
    }






    val finalStep = { weights: List<Double> ->
        recalc(weights)
        val bm1 = baseMean * weights[0]
        val bm2 = baseMean2 * weights[1]
        val bsd1 = baseSd * weights[0]
        val bsd2 = baseSd2 * weights[1]

        var averageSecondMoment = 0.0
        var averageMeanSquared = 0.0
        var s1 = 0.0
        var s2 = 0.0

        s1 += weights[0] * (baseMean.pow(2.0) + baseSd.pow(2.0))
        s1 += weights[1] * (baseMean2.pow(2.0) + baseSd2.pow(2.0))
        s2 += bm1 + bm2
        s2 = s2.pow(2.0)


        averageSecondMoment += weights[0] * secondMoment1
        averageSecondMoment += weights[1] * secondMoment2

        averageMeanSquared += weights[0] * baseMean
        averageMeanSquared += weights[1] * baseMean2
        averageMeanSquared = averageMeanSquared.pow(2.0)




        val finalVariance = averageSecondMoment - averageMeanSquared
        val finalSd = Math.sqrt(finalVariance)
        val finalMean = bm1 + bm2

//        val z1 = scores.map { (it - baseMean) / baseSd }
//        val z2 = scores2.map { (it - baseMean2) / baseSd2 }
//
//        val newScores = z1.zip(z2).map { it.first * weights[0] + it.second * weights[1] }



//        val newScores = scores.zip(scores2).map { it.first * weights[0] + it.second * weights[1] }
//        val newSecond = newScores.map { it.pow(2.0) }.average() - (newScores.average()).pow(2.0)
//        println(newSecond.pow(0.5))
//        println(myDistCombined.sd)
//        val final = newScores.map { it * finalSd + finalMean }
//        myDistCombined.recalculateDist(final)
        println("RESULT: ${myDistCombined.mean} : ${myDistCombined.sd}")
        println(finalSd)
        println(Math.sqrt(s1 - s2))



//        reportResult(myDist, bm1, bsd1, "d1")
//        reportResult(myDist2, bm2, bsd2, "d2")
//        reportResult(myDistCombined, bm1 + bm2, bsd1 + bsd2, "combined")
//        println("Final sd: $finalSd , Final Variance: $finalVariance")
    }


    finalStep(listOf(0.3, 0.7))
//    println("Dist1 Sim: ${myDist.sim(0.2)}")
//    println("Dist2 Sim: ${myDist2.sim(0.2)}")
//    println("Dist3 Sim: ${myDistCombined.sim(0.2)}")

//    finalStep(listOf(0.5, 0.5))


//    with (myDist) {
//        println("$mean (${baseMean * 0.4}): $sd (${baseSd * 0.4}) ")
//    }
//
//    myDist.recalculateDist(scores.map { it * 1.5 })
//
//    with (myDist) {
//        println("$mean (${baseMean * 1.5}): $sd (${baseSd * 1.5}) ")
//    }
}

