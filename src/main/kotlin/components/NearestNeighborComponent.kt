package components

import containers.EmailSparseVector
import kernels.GaussianKernel
import kernels.LaplacianKernel
import kernels.SimilarityFuns


class NearestNeighborComponent(var k: Int,
                               val trainingVectors: List<EmailSparseVector>) {

//    val kernel = LaplacianKernel(SimilarityFuns::simComponentL1Dist)
    val mySim = { v1: EmailSparseVector, v2: EmailSparseVector ->
    SimilarityFuns.simComponentL1DistWeights(v1.components, v2.components, listOf(0.6501437717888071, 0.9598502751788944, 0.32039458320494574, 0.6889872455867913, 0.22792011655853872, 0.7093149923370305, 0.7910443461701647, 0.8216745106053818, 0.4024084348316237, 0.2992834776742689, 0.6995217376468832, 0.05517870230870561, 0.7436039904549169, 0.183914586934225, 0.8833427764918287, 0.16209955659871844, 0.5742888929387173, 0.3823620785435912, 0.7564431184425912, 0.37679333168890256))
//    SimilarityFuns.dotProduct(v1, listOf(0.21171639189969188, 0.8815089671944092, 0.08452820268581479, 0.9692908051041536, 0.06832001305990681, 0.20630014652905046, 0.586042915639428, 0.07967565539266346, 0.8346004178551936, 0.4010602026399084, -0.06294940994115016, 0.7776141981446836, 0.09722908703191285, 0.9712069052086312, 0.19821409163856443, 0.38521042944457207, 0.774729341696385, 0.395928717548814, 0.8487041753926144, 0.11351952143964586) + 0.30417410688366275)
}
//    fun classify(unknown: EmailSparseVector): String = if (mySim(unknown, unknown) > 0.0) "spam" else "ham"

    fun classify(unknown: EmailSparseVector) =
        trainingVectors
//            .map { v -> v to SimilarityFuns.simComponentL1Dist(v, unknown) }
//            .map { v -> v to kernel.sim(v, unknown) }
            .map { v -> v to mySim(v, unknown) }
//            .map { v -> v to kernel.sim(v, unknown) }
//            .map { v -> v to SimilarityFuns.simComponentCosine(v, unknown) }
            .sortedBy { it.second }
//            .sortedByDescending { it.second }
            .take(k) //KNN
//            .run {
//                val total = this.sumByDouble { 1.0 / it.second }
////                val total = this.sumByDouble { it.second }
//                val normalized = this.map { it.first to (1.0 / it.second) / total }
////                val normalized = this.map { it.first to (it.second) / total }
//                val final = normalized.sumByDouble {
//                    val lScore = if (it.first.label == "ham") 1.0 else 0.0
//                    lScore * it.second
//                }
//                if (final > 0.5) "ham" else "spam"
//            }
            .map { it.first.label }
            .groupingBy { it }
            .eachCount()
            .maxBy { it.value }!!.key
}
