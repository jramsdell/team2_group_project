package kernels

import containers.EmailSparseVector
import kotlin.math.exp
import kotlin.math.pow


class GaussianKernel(similarityFun: (Map<String, Double>, Map<String, Double>) -> Double,
                     val sigma: Double = 2.0) : KernelBase(similarityFun) {

    override fun sim(v1: Map<String,Double>, v2: Map<String,Double>): Double {
        val result = super.sim(v1, v2)
        return exp(-(result.pow(2.0) / (2 * sigma.pow(2.0))))
    }

}