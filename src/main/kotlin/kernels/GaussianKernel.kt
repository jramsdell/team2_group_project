package kernels

import containers.EmailSparseVector
import kotlin.math.exp
import kotlin.math.pow


class GaussianKernel(similarityFun: (EmailSparseVector, EmailSparseVector) -> Double,
                     val sigma: Double = 2.0) : KernelBase(similarityFun) {

    override fun sim(v1: EmailSparseVector, v2: EmailSparseVector): Double {
        val result = super.sim(v1, v2)
        return exp(-(result.pow(2.0) / (2 * sigma.pow(2.0))))
    }

}