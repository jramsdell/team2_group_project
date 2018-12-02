package kernels

import containers.EmailSparseVector
import utils.sigmoid
import kotlin.math.exp
import kotlin.math.pow


class SigmoidKernel(similarityFun: (Map<String, Double>, Map<String, Double>) -> Double) : KernelBase(similarityFun) {

    override fun sim(v1: Map<String,Double>, v2: Map<String,Double>): Double {
        val result = super.sim(v1, v2)
        return result.sigmoid()
    }

}