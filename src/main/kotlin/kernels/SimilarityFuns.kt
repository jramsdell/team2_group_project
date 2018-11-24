package kernels

import containers.EmailSparseVector
import utils.toHashMap
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

    fun simOverlap(v1: EmailSparseVector, v2: EmailSparseVector): Double {
        val keys = v1.components.keys.intersect(v2.components.keys)
        return (2 * keys.size.toDouble()) / (v1.components.keys.size + v2.components.keys.size)
    }

    fun dotProduct(v1: EmailSparseVector, weights: List<Double>): Double {
        val keys = v1.components.keys
        return keys.sumByDouble { key ->
            val v1Component = v1.components[key] ?: 0.0
            (v1Component * weights[key.toInt()])
        }
    }

    fun dotProducts(v1: EmailSparseVector, weights: List<Double>, partitions: Map<Int, Int>): HashMap<Int, Double> {
        val bins = partitions.values.map { it to 0.0 }.toHashMap()
        val keys = v1.components.keys
        keys.forEach { key ->
            val v1Component = v1.components[key] ?: 0.0
            val loc = partitions[key.toInt()]!!
            bins[loc] = bins[loc]!! + v1Component * weights[key.toInt()]
        }
        return bins
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
}