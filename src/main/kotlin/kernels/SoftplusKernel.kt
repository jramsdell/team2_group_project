package kernels

import containers.EmailSparseVector
import org.apache.commons.math3.distribution.NormalDistribution
import utils.sigmoid
import utils.tanh
import kotlin.math.absoluteValue
import kotlin.math.exp
import kotlin.math.pow


class SoftplusKernel(similarityFun: (Map<String, Double>, Map<String, Double>) -> Double) : KernelBase(similarityFun) {
//    val dist = NormalDistribution(0.0, 0.005)

    override fun sim(v1: Map<String,Double>, v2: Map<String,Double>): Double {
        val result = super.sim(v1, v2)
        return Math.log(1.0 +  Math.exp(result))
    }

}