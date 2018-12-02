package predictors

import components.StochasticComponent
import components.TrainingVectorComponent
import containers.EmailSparseVector
import edu.unh.cs753.predictors.LabelPredictor
import edu.unh.cs753.utils.SearchUtils
import learning.training.SimpleStochasticTrainer
import org.apache.lucene.search.IndexSearcher

class CombinedBasisPredictor(searcher: IndexSearcher) : LabelPredictor(searcher) {
    val trainer = SimpleStochasticTrainer(searcher)
    var labeler = trainer.doTrain2()

    override fun predict(tokens: MutableList<String>?): String {
        val dist = tokens!!
            .flatMap { trainer.trainingComponent.createCharacterGrams(it, 4) }
            .groupingBy { it }
            .eachCount()
            .map { it.key to it.value.toDouble() }
            .toMap()

        val dist2 = tokens!!
            .run { trainer.trainingComponent.createBigrams(this) }
            .groupingBy { it }
            .eachCount()
            .map { it.key to it.value.toDouble() }
            .toMap()


        val v = EmailSparseVector("", components = dist, bigrams = dist2)
        return labeler(v)
    }


}

fun main(args: Array<String>) {
    val searcher = SearchUtils.createIndexSearcher("index")
    val predictor = CombinedBasisPredictor(searcher)
//    predictor.stochastic.doTrain()
    predictor.evaluate()


}
