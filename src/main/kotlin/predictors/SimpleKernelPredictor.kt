package predictors

import components.NearestNeighborComponent
import components.StochasticComponent
import components.TrainingVectorComponent
import containers.EmailSparseVector
import edu.unh.cs753.predictors.LabelPredictor
import edu.unh.cs753.utils.SearchUtils
import org.apache.lucene.search.IndexSearcher


class SimpleKernelPredictor(searcher: IndexSearcher) : LabelPredictor(searcher) {
    val trainingComponent = TrainingVectorComponent(searcher)
//    val stochastic = StochasticComponent(trainingComponent.basisVectors, trainingComponent.vectors)
    val knn = NearestNeighborComponent(k = 3, trainingVectors = trainingComponent.vectors)
//    val centroid = CentroidComponent(5, trainingVectors = trainingComponent.vectors.toList())


    override fun predict(tokens: MutableList<String>?): String {
        val dist = tokens!!.groupingBy { it }
            .eachCount()
            .map { it.key to it.value.toDouble() }
            .toMap()

        val v = EmailSparseVector("", dist)
        val embedding = trainingComponent.embed(v)

        return knn.classify(embedding)
//        return "spam"
//        return centroid.classify(embedding)
    }

}

fun main(args: Array<String>) {
    val searcher = SearchUtils.createIndexSearcher("index")
    val predictor = SimpleKernelPredictor(searcher)
//    predictor.stochastic.doTrain()
    predictor.evaluate(1000)


}