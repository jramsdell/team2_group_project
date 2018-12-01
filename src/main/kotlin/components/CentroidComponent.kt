package components

import containers.EmailSparseVector
import kernels.LaplacianKernel
import kernels.SimilarityFuns


//class CentroidComponent(var k: Int,
//                         trainingVectors: List<EmailSparseVector>) {
//
//    val spamCount = trainingVectors.count { it.label == "spam" }
//    val hamCount = trainingVectors.size - spamCount
////    val kernel = LaplacianKernel(SimilarityFuns::simComponentCosine)
//
//    val spamCentroid = trainingVectors
//        .filter { it.label == "spam" }
//        .flatMap { it.components.entries.map { it.toPair() } }
//        .groupBy { it.first }
//        .map { it.key to it.value.sumByDouble { it.second } / spamCount }
//        .toMap()
//        .run { EmailSparseVector(label = "spam", components = this) }
//
//    val hamCentroid = trainingVectors
//        .filter { it.label == "ham" }
//        .flatMap { it.components.entries.map { it.toPair() } }
//        .groupBy { it.first }
//        .map { it.key to it.value.sumByDouble { it.second } / hamCount }
//        .toMap()
//        .run { EmailSparseVector(label = "ham", components = this) }
//
//    fun classify(unknown: EmailSparseVector): String {
//        val hamScore = SimilarityFuns.simComponentCosine(hamCentroid.components, unknown.components)
//        val spamScore = SimilarityFuns.simComponentCosine(spamCentroid.components, unknown.components)
//        return if (hamScore > spamScore) "ham" else "spam"
//    }
//}
