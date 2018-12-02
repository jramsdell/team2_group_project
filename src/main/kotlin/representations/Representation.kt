package representations

import kernels.SimilarityFuns
import kernels.SoftplusKernel

typealias ComponentKey = Pair<RepresentationType, KernelType>


data class Representation (val components: Map<String, Double>, val similarityFunction: (Representation, Representation) -> Double,
                                                                 val key: ComponentKey, var score: Double = 0.0, var curScore: Double = 0.0) {
    fun sim(other: Representation): Double = similarityFunction(this, other)


    companion object {

        fun SIM_COSINE(rep1: Representation, rep2: Representation): Double =
             SimilarityFuns.softPlus(SimilarityFuns.simComponentCosine(rep1.components, rep2.components))

        fun SIM_DOT(rep1: Representation, rep2: Representation): Double =
                SimilarityFuns.softPlus(SimilarityFuns.simComponentDot(rep1.components, rep2.components))

        fun SIM_OVERLAP(rep1: Representation, rep2: Representation): Double =
                SimilarityFuns.softPlus(SimilarityFuns.simOverlap(rep1.components, rep2.components))

    }

}

