package predictors

import components.NearestNeighborComponent
import components.StochasticComponent
import components.TrainingVectorComponent
import containers.EmailSparseVector
import edu.unh.cs753.predictors.LabelPredictor
import edu.unh.cs753.utils.SearchUtils
import org.apache.lucene.search.IndexSearcher


//class SimpleKernelPredictor(searcher: IndexSearcher) : LabelPredictor(searcher) {
//    val trainingComponent = TrainingVectorComponent(searcher)
////    val stochastic = StochasticComponent(trainingComponent.basisVectors, trainingComponent.vectors)
//    val knn = NearestNeighborComponent(k = 3, trainingVectors = trainingComponent.vectors)
////    val centroid = CentroidComponent(5, trainingVectors = trainingComponent.vectors.toList())
//    val stochastic = StochasticComponent(trainingComponent.nBases, trainingComponent.vectors, trainingComponent.holdout)
//
//
//    override fun predict(tokens: MutableList<String>?): String {
//        val dist = tokens!!.groupingBy { it }
//            .eachCount()
//            .map { it.key to it.value.toDouble() }
//            .toMap()
//
//        val v = EmailSparseVector("", dist)
//        val embedding = trainingComponent.embed(v)
//        val weights = listOf(-2.0809679517532507, -2.090643168931246, 0.8838631908037125, -3.728043746748096, 1.269391347346216, 5.399407500057859, 1.0361073064170885, 2.2545661461069195, 0.5558237604761755, -2.8769468359852977)
//
//        val result = stochastic.myLabeler(weights, emptyList())(embedding)
//        return result
//
////        return knn.classify(embedding)
////        return "spam"
////        return centroid.classify(embedding)
//    }
////    -2.0809679517532507, -2.090643168931246, 0.8838631908037125, -3.728043746748096, 1.269391347346216, 5.399407500057859, 1.0361073064170885, 2.2545661461069195, 0.5558237604761755, -2.8769468359852977
//
//
//}
//
//fun main(args: Array<String>) {
//    val searcher = SearchUtils.createIndexSearcher("index")
//    val predictor = SimpleKernelPredictor(searcher)
////    predictor.stochastic.doTrain()
//    predictor.evaluate(1000)
//
//
//}