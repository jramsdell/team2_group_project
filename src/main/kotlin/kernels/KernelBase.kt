package kernels

import containers.EmailSparseVector


abstract class KernelBase(private val similarityFun: (Map<String,Double>, Map<String, Double>) -> Double) {
     open fun sim(v1: Map<String, Double>, v2: Map<String, Double>): Double = similarityFun(v1, v2)
}