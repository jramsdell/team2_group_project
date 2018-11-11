package kernels

import containers.EmailSparseVector


abstract class KernelBase(private val similarityFun: (EmailSparseVector, EmailSparseVector) -> Double) {
     open fun sim(v1: EmailSparseVector, v2: EmailSparseVector): Double = similarityFun(v1, v2)
}