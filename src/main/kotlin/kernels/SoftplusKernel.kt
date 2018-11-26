package kernels

import containers.EmailSparseVector
import org.apache.commons.math3.distribution.NormalDistribution
import utils.sigmoid
import utils.tanh
import kotlin.math.absoluteValue
import kotlin.math.exp
import kotlin.math.pow


class SoftplusKernel(similarityFun: (EmailSparseVector, EmailSparseVector) -> Double) : KernelBase(similarityFun) {
//    val dist = NormalDistribution(0.0, 0.005)

    override fun sim(v1: EmailSparseVector, v2: EmailSparseVector): Double {
        val result = super.sim(v1, v2)
//        return Math.log(1.0 + Math.exp(result .run { this + this * dist.sample().absoluteValue }))
//        return Math.log(1.0 + Math.exp(result)).run { this + this * dist.sample().absoluteValue }
        return Math.log(1.0 +  Math.exp(result))
    }

}