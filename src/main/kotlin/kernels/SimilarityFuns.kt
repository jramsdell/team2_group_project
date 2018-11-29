package kernels

import containers.EmailSparseVector
import info.debatty.java.stringsimilarity.*
import utils.*
import kotlin.math.absoluteValue
import kotlin.math.pow

object SimilarityFuns {

    fun simComponentL1Dist(v1: EmailSparseVector, v2: EmailSparseVector): Double {
        val keys = v1.components.keys.union(v2.components.keys)
        return keys.sumByDouble { key ->
            val v1Component = v1.components[key] ?: 0.0
            val v2Component = v2.components[key] ?: 0.0
            (v1Component - v2Component).absoluteValue
        }
    }

    fun simComponentRawDiff(v1: EmailSparseVector, v2: EmailSparseVector): Double {
        val keys = v1.components.keys.union(v2.components.keys)
        return keys.sumByDouble { key ->
            val v1Component = v1.components[key] ?: 0.0
            val v2Component = v2.components[key] ?: 0.0
            (v1Component - v2Component)
        }
    }

    fun simComponentDot(v1: EmailSparseVector, v2: EmailSparseVector): Double {
        val keys = v1.components.keys.union(v2.components.keys)
        val v1Norm = v1.components.normalize()
        val v2Norm = v2.components.normalize()

        return keys.sumByDouble { key ->
            val v1Component = v1Norm[key] ?: 0.0
            val v2Component = v2Norm[key] ?: 0.0
            v1Component * v2Component
        }
    }

    fun simBigramDot(v1: EmailSparseVector, v2: EmailSparseVector): Double {
        val keys = v1.bigrams.keys.union(v2.bigrams.keys)
        val v1Norm = v1.bigrams.normalize()
        val v2Norm = v2.bigrams.normalize()

        return keys.sumByDouble { key ->
            val v1Component = v1Norm[key] ?: 0.0
            val v2Component = v2Norm[key] ?: 0.0
            v1Component * v2Component
        }
    }

    fun simComponentNonlinear(v1: EmailSparseVector, v2: EmailSparseVector): Double {
        val keys = v1.components.keys.union(v2.components.keys)
        val v1Norm = v1.components.normalize()
        val v2Norm = v2.components.normalize()

        return keys.sumByDouble { key ->
            val v1Component = v1Norm[key] ?: 0.0
            val v2Component = v2Norm[key] ?: 0.0
            v1Component.pow(2.0) * v2Component.pow(2.0)
        }
    }

    fun simComponentDotCovariance(v1: EmailSparseVector, v2: EmailSparseVector, covarianceMap: HashMap<String, HashMap<String, Double>>): Double {
//        val keys = v1.components.keys.union(v2.components.keys)
//        val v1Norm = v1.components.takeMostFrequent(5)
//        val v2Norm = v2.components.takeMostFrequent(5)

        var score = 0.0

        v1.components.forEach { (k,v) ->
            val dist = covarianceMap[k]
            if (dist != null) {
                v2.components.forEach { (k2, v2) ->
                    score +=  (dist[k2] ?: 0.0)
                }
            }
        }


        return score

    }

    fun simComponentString(v1: EmailSparseVector, v2: EmailSparseVector): Double {
        val sim = NormalizedLevenshtein()

        val best1 = v1.components.entries.toList()
            .sortedByDescending { it.value }.take(10)
        val best2 = v2.components.entries.toList()
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

    fun simComponentDotKld(v1: EmailSparseVector, v2: EmailSparseVector): Double {
        val keys = v1.components.keys.union(v2.components.keys)
        val v1Norm = v1.components.normalize()
        val v2Norm = v2.components.normalize()

        return keys.sumByDouble { key ->
            val v1Component = v1Norm[key] ?: 0.0
            val v2Component = v2Norm[key] ?: 0.0
            val k1 = v1Component * Math.log(v1Component / v2Component).defaultWhenNotFinite(0.0)
            val k2 = v2Component * Math.log(v2Component / v1Component).defaultWhenNotFinite(0.0)
            ((k1 + k2) / 2.0).defaultWhenNotFinite(0.0)
        }
    }


    fun simOverlap(v1: EmailSparseVector, v2: EmailSparseVector): Double {
        val keys = v1.components.keys.intersect(v2.components.keys)
        return (2 * keys.size.toDouble()) / (v1.components.keys.size + v2.components.keys.size)
    }

    fun simBigramOverlap(v1: EmailSparseVector, v2: EmailSparseVector): Double {
        val keys = v1.bigrams.keys.intersect(v2.bigrams.keys)
        return (2 * keys.size.toDouble()) / (v1.bigrams.keys.size + v2.bigrams.keys.size)
    }

    fun dotProduct(v1: EmailSparseVector, weights: List<Double>): Double {
        val keys = v1.components.keys
        return keys.sumByDouble { key ->
            val v1Component = v1.components[key] ?: 0.0
            (v1Component * weights[key.toInt()])
        }
    }




    fun simComponentL1DistWeights(v1: EmailSparseVector, v2: EmailSparseVector, weights: List<Double>): Double {
        val keys = v1.components.keys.union(v2.components.keys)
        return keys.sumByDouble { key ->
            val v1Component = v1.components[key] ?: 0.0
            val v2Component = v2.components[key] ?: 0.0
            (v1Component - v2Component).absoluteValue * weights[key.toInt()]
        }
    }

    fun simComponentL2Dist(v1: EmailSparseVector, v2: EmailSparseVector): Double {
        val keys = v1.components.keys.union(v2.components.keys)
        return keys.sumByDouble { key ->
            val v1Component = v1.components[key] ?: 0.0
            val v2Component = v2.components[key] ?: 0.0
            (v1Component - v2Component).pow(2.0)
        }.pow(0.5)
    }

    fun simComponentL2DistWeights(v1: EmailSparseVector, v2: EmailSparseVector, weights: List<Double>): Double {
        val keys = v1.components.keys.union(v2.components.keys)
        return keys.sumByDouble { key ->
            val v1Component = v1.components[key] ?: 0.0
            val v2Component = v2.components[key] ?: 0.0
            (v1Component - v2Component).pow(2.0) * weights[key.toInt()]
        }.pow(0.5)
    }

    fun simComponentCosine(v1: EmailSparseVector, v2: EmailSparseVector): Double {
        val keys = v1.components.keys.union(v2.components.keys)
        val dotProduct =  keys.sumByDouble { key ->
            val v1Component = v1.components[key] ?: 0.0
            val v2Component = v2.components[key] ?: 0.0
            v1Component * v2Component
        }

        val v1Norm = v1.components.values.sumByDouble { it.pow(2) }.pow(0.5)
        val v2Norm = v2.components.values.sumByDouble { it.pow(2) }.pow(0.5)

        return dotProduct / (v1Norm * v2Norm)
    }


    fun simBigramCosine(v1: EmailSparseVector, v2: EmailSparseVector): Double {
        val keys = v1.bigrams.keys.union(v2.bigrams.keys)
        val dotProduct =  keys.sumByDouble { key ->
            val v1Component = v1.bigrams[key] ?: 0.0
            val v2Component = v2.bigrams[key] ?: 0.0
            v1Component * v2Component
        }

        val v1Norm = v1.bigrams.values.sumByDouble { it.pow(2) }.pow(0.5)
        val v2Norm = v2.bigrams.values.sumByDouble { it.pow(2) }.pow(0.5)

        return dotProduct / (v1Norm * v2Norm)
    }
}