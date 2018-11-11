package learning.training

import components.StochasticComponent
import components.TrainingVectorComponent
import edu.unh.cs753.utils.SearchUtils
import org.apache.lucene.search.IndexSearcher


class SimpleStochasticTrainer(val searcher: IndexSearcher) {
    val trainingComponent = TrainingVectorComponent(searcher)
    val stochastic = StochasticComponent(trainingComponent.basisVectors, trainingComponent.vectors, trainingComponent)

}

fun main(args: Array<String>) {
    val searcher = SearchUtils.createIndexSearcher("index")
    val predictor = SimpleStochasticTrainer(searcher)
    predictor.stochastic.doTrain()
}