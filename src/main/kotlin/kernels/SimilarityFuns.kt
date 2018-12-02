package kernels

import containers.EmailSparseVector
import info.debatty.java.stringsimilarity.*
import utils.*
import java.lang.Double.sum
import kotlin.math.absoluteValue
import kotlin.math.pow

object SimilarityFuns {

    fun simComponentL1Dist(v1: Map<String,Double>, v2: Map<String,Double>): Double {
        val keys = v1.keys.union(v2.keys)
        return keys.sumByDouble { key ->
            val v1Component = v1[key] ?: 0.0
            val v2Component = v2[key] ?: 0.0
            (v1Component - v2Component).absoluteValue
        }
    }

    fun simComponentRawDiff(v1: Map<String,Double>, v2: Map<String,Double>): Double {
        val keys = v1.keys.union(v2.keys)
        return keys.sumByDouble { key ->
            val v1Component = v1[key] ?: 0.0
            val v2Component = v2[key] ?: 0.0
            (v1Component - v2Component)
        }
    }

    fun simComponentDot(v1: Map<String,Double>, v2: Map<String,Double>): Double {
        val keys = v1.keys.union(v2.keys)
        val v1Norm = v1
        val v2Norm = v2

        return keys.sumByDouble { key ->
            val v1Component = v1Norm[key] ?: 0.0
            val v2Component = v2Norm[key] ?: 0.0
            v1Component * v2Component
        }
    }

    fun simBigramDot(v1: Map<String,Double>, v2: Map<String,Double>): Double {
        val keys = v1.keys.union(v2.keys)
        val v1Norm = v1
        val v2Norm = v2

        return keys.sumByDouble { key ->
            val v1Component = v1Norm[key] ?: 0.0
            val v2Component = v2Norm[key] ?: 0.0
            v1Component * v2Component
        }
    }

    fun simComponentNonlinear(v1: Map<String,Double>, v2: Map<String,Double>): Double {
        val keys = v1.keys.union(v2.keys)
        val v1Norm = v1.normalize()
        val v2Norm = v2.normalize()

        return keys.sumByDouble { key ->
            val v1Component = v1Norm[key] ?: 0.0
            val v2Component = v2Norm[key] ?: 0.0
            v1Component.pow(2.0) * v2Component.pow(2.0)
        }
    }

    private fun symKldDist(d1: List<Double>, d2: List<Double>): Double = d1.zip(d2).sumByDouble { (v1, v2) ->
        val div1 = (v1 * Math.log(v1) + v2 * Math.log(v2))
        val div2 = (v1 + v2) * Math.log(v1 + v2)
        div1 - div2
    }

    fun simComponentDotCovariance(v1: Map<String,Double>, v2: Map<String,Double>, covarianceMap: HashMap<String, HashMap<String, Double>>): Double {
        var score = 0.0
        var score2 = 0.0
        val keys = v1.keys.union(v2.keys)
//        keys.forEach {  k1 ->
//            val dist = covarianceMap[k1]
//            if (dist != null) {
//                keys.forEach { k2 ->
//                    score += (dist[k2] ?: 0.0)
//                }
//            }
//
//        }

        v1.forEach { (k,vd1) ->
            val dist = covarianceMap[k]
            if (dist != null) {
                v2.forEach { (k2, vd2) ->
                    score +=  (dist[k2] ?: 0.0)
                }
            }
        }



        return score
    }

    fun simComponentDotCovariance2(v1: Map<String,Double>, v2: Map<String,Double>, covarianceMap: HashMap<String, HashMap<String, Double>>): Double {
        var score = 0.0
        val v1Dist = HashMap<String, Double>()
        val v2Dist = HashMap<String, Double>()


        v1.forEach { (k,vd1) ->
            val dist = covarianceMap[k]
            dist?.forEach { (neighbor, score) ->
                v1Dist.merge(neighbor, score * vd1, ::sum)
            }
        }




        v2.forEach { (k,vd1) ->
            val dist = covarianceMap[k]
            dist?.forEach { (neighbor, score) ->
                v2Dist.merge(neighbor, score * vd1, ::sum)
            }
        }


        val v1DistFinal = v1Dist.normalize()
        val v2DistFinal = v2Dist.normalize()

        v1DistFinal.keys.intersect(v2DistFinal.keys).forEach { k ->
            val vR1 = v1DistFinal[k]!!.defaultWhenNotFinite(1/v1DistFinal.keys.size.toDouble())
            val vR2 = v2DistFinal[k]!!.defaultWhenNotFinite(1/v2DistFinal.keys.size.toDouble())
            val div1 = (vR1 * Math.log(vR1) + vR2 * Math.log(vR2))
            val div2 = (vR1+ vR2) * Math.log(vR1 + vR2)
            score += (div1 * div2).defaultWhenNotFinite(0.0)
        }

        return score

    }

//    fun myKld(d1: HashMap<String, Double>, d2: HashMap<String, Double>): Double {
//        d1.keys.intersect(d2.keys)
//            .forEach { () }
//    }


    fun simComponentString(v1: Map<String,Double>, v2: Map<String,Double>): Double {
        val sim = NormalizedLevenshtein()

        val best1 = v1.entries.toList()
            .sortedByDescending { it.value }.take(10)
        val best2 = v2.entries.toList()
            .sortedByDescending { it.value }.take(10)


        val results = best1.map{ (v1Comp, v1Freq) ->
            best2.map { (v2Comp, v2Freq) ->
                sim.similarity(v1Comp, v2Comp) * Math.log(v1Freq) * Math.log(v2Freq)
            }.max()!!
        }.max()!!

        val results2 = best2.map{ (v1Comp, v1Freq) ->
            best1.map { (v2Comp, v2Freq) ->
                sim.similarity(v1Comp, v2Comp) * Math.log(v1Freq) * Math.log(v2Freq)
            }.max()!!
        }.max()!!

        return results.defaultWhenNotFinite(0.0) + results2.defaultWhenNotFinite(0.0)
    }

//    private fun kld(d1: List<Double>, d2: List<Double>): Double = d1.zip(d2).sumByDouble { (v1, v2) ->
//        v1 * Math.log(v1 / v2)
//    }
//
//    private fun symKldDist3(d1: List<Double>, d2: List<Double>): Double {
//        val midpoint = d1.zip(d2).map { it.first * 0.5 + it.second * 0.5 }
//        return kld(d1, midpoint) * 0.5 + kld(d2, midpoint) * 0.5
//    }

    fun simComponentDotKld(v1: Map<String,Double>, v2: Map<String,Double>): Double {
        val keys = v1.keys.union(v2.keys)
        val v1Norm = v1.normalize()
        val v2Norm = v2.normalize()

        return keys.sumByDouble { key ->
            val v1Component = v1Norm[key] ?: 0.0
            val v2Component = v2Norm[key] ?: 0.0
            val vMid = v1Component * 0.5 + v2Component * 0.5
            val k1 = v1Component * Math.log(v1Component / vMid).defaultWhenNotFinite(0.0)
            val k2 = v2Component * Math.log(v2Component / vMid).defaultWhenNotFinite(0.0)
            ((k1 + k2) / 2.0).defaultWhenNotFinite(0.0)
        }
    }

    fun simComponentDotKldBigram(v1: Map<String,Double>, v2: Map<String,Double>): Double {
        val keys = v1.keys.union(v2.keys)
        val v1Norm = v1.normalize()
        val v2Norm = v2.normalize()

        return keys.sumByDouble { key ->
            val v1Component = v1Norm[key] ?: 0.0
            val v2Component = v2Norm[key] ?: 0.0
            val vMid = v1Component * 0.5 + v2Component * 0.5
            val k1 = v1Component * Math.log(v1Component / vMid).defaultWhenNotFinite(0.0)
            val k2 = v2Component * Math.log(v2Component / vMid).defaultWhenNotFinite(0.0)
            ((k1 + k2) / 2.0).defaultWhenNotFinite(0.0)
        }
    }


    fun simOverlap(v1: Map<String,Double>, v2: Map<String,Double>): Double {
        val keys = v1.keys.intersect(v2.keys)
        return (2 * keys.size.toDouble()) / (v1.keys.size + v2.keys.size)
    }

    fun simBigramOverlap(v1: Map<String,Double>, v2: Map<String,Double>): Double {
        val keys = v1.keys.intersect(v2.keys)
        return (2 * keys.size.toDouble()) / (v1.keys.size + v2.keys.size)
    }

    fun dotProduct(v1: Map<String,Double>, weights: List<Double>): Double {
        val keys = v1.keys
        return keys.sumByDouble { key ->
            val v1Component = v1[key] ?: 0.0
            (v1Component * weights[key.toInt()])
        }
    }




    fun simComponentL1DistWeights(v1: Map<String,Double>, v2: Map<String,Double>, weights: List<Double>): Double {
        val keys = v1.keys.union(v2.keys)
        return keys.sumByDouble { key ->
            val v1Component = v1[key] ?: 0.0
            val v2Component = v2[key] ?: 0.0
            (v1Component - v2Component).absoluteValue * weights[key.toInt()]
        }
    }

    fun simComponentL2Dist(v1: Map<String,Double>, v2: Map<String,Double>): Double {
        val keys = v1.keys.union(v2.keys)
        return keys.sumByDouble { key ->
            val v1Component = v1[key] ?: 0.0
            val v2Component = v2[key] ?: 0.0
            (v1Component - v2Component).pow(2.0)
        }.pow(0.5)
    }

    fun simComponentL2DistWeights(v1: Map<String,Double>, v2: Map<String,Double>, weights: List<Double>): Double {
        val keys = v1.keys.union(v2.keys)
        return keys.sumByDouble { key ->
            val v1Component = v1[key] ?: 0.0
            val v2Component = v2[key] ?: 0.0
            (v1Component - v2Component).pow(2.0) * weights[key.toInt()]
        }.pow(0.5)
    }

    fun simComponentCosine(v1: Map<String,Double>, v2: Map<String,Double>): Double {
        val keys = v1.keys.union(v2.keys)
        val dotProduct =  keys.sumByDouble { key ->
            val v1Component = v1[key] ?: 0.0
            val v2Component = v2[key] ?: 0.0
            v1Component * v2Component
        }

        val v1Norm = v1.values.sumByDouble { it.pow(2) }.pow(0.5)
        val v2Norm = v2.values.sumByDouble { it.pow(2) }.pow(0.5)

        return dotProduct / (v1Norm * v2Norm)
    }


    fun simBigramCosine(v1: Map<String,Double>, v2: Map<String,Double>): Double {
        val keys = v1.keys.union(v2.keys)
        val dotProduct =  keys.sumByDouble { key ->
            val v1Component = v1[key] ?: 0.0
            val v2Component = v2[key] ?: 0.0
            v1Component * v2Component
        }

        val v1Norm = v1.values.sumByDouble { it.pow(2) }.pow(0.5)
        val v2Norm = v2.values.sumByDouble { it.pow(2) }.pow(0.5)

        return dotProduct / (v1Norm * v2Norm)
    }

    fun softPlus(x: Double) = Math.log(Math.exp(x) + 1.0)
}