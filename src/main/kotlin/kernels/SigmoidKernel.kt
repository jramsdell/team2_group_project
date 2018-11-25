package kernels

import containers.EmailSparseVector
import utils.sigmoid
import kotlin.math.exp
import kotlin.math.pow


class SigmoidKernel(similarityFun: (EmailSparseVector, EmailSparseVector) -> Double) : KernelBase(similarityFun) {

    override fun sim(v1: EmailSparseVector, v2: EmailSparseVector): Double {
        val result = super.sim(v1, v2)
        return result.sigmoid()
    }

}