package learning

import components.TrainingVectorComponent
import edu.unh.cs753.utils.SearchUtils
import kernels.SimilarityFuns
import org.apache.lucene.search.IndexSearcher
import java.util.*


class KDTester(val searcher: IndexSearcher) {
    val trainingComponent = TrainingVectorComponent(searcher)
    val ghettoKDTree = GhettoKDTree(trainingComponent)

    fun testHoldout() {
//        val holdouts = trainingComponent.holdout[Random(21230).nextInt(trainingComponent.holdout.size)]
        val holdouts = trainingComponent.holdout.take(100)

        holdouts.forEach { holdout ->
            val nearest = trainingComponent.vectors.map { bv ->
                bv to SimilarityFuns.simComponentL1Dist(holdout, bv)
            }
                .sortedBy { it.second }
                .withIndex()
                .toList()

            val predictedNearest = ghettoKDTree.retrieveCandidate(holdout)

            val trueIndex = nearest.find { it.value.first == predictedNearest }!!.index
            println("Predicted: $trueIndex")
        }


    }
}

fun main(args: Array<String>) {
    val searcher = SearchUtils.createIndexSearcher("index")
    val tester = KDTester(searcher)
//    tester.trainingComponent.basisVectors.forEach { bv -> println(bv.id) }
    val index = 2
    println(tester.trainingComponent.basisVectors[index].id)
    tester.ghettoKDTree.getDerivative(index).forEach { println(it) }
//    tester.testHoldout()

}
