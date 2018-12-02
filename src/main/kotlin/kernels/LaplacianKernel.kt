package kernels

import containers.EmailSparseVector
import kotlin.math.exp
import kotlin.math.pow


class LaplacianKernel(similarityFun: (Map<String, Double>, Map<String, Double>) -> Double,
                      val sigma: Double = 1.5) : KernelBase(similarityFun) {

    override fun sim(v1: Map<String,Double>, v2: Map<String,Double>): Double {
        val result = super.sim(v1, v2)
        return exp(-(result / (sigma)))
    }

}