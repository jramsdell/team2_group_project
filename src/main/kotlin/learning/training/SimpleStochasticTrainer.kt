package learning.training

import components.StochasticComponent
import components.TrainingVectorComponent
import containers.EmailSparseVector
import edu.unh.cs753.utils.SearchUtils
import learning.GhettoKDTree
import org.apache.lucene.search.IndexSearcher
import utils.*
import java.lang.Double.sum
import kotlin.math.ln
import kotlin.math.pow


class SimpleStochasticTrainer(val searcher: IndexSearcher) {
    val trainingComponent = TrainingVectorComponent(searcher)

    fun convertResult(weights: List<Double>, emails: List<EmailSparseVector>): EmailSparseVector {
        val finalComponents = HashMap<String, Double>()
        val wNorm = weights.sumByDouble { it.pow(2.0) }
        emails.zip(weights).forEach { (e, weight) ->
            if (weight != 0.0) {
                val vNorm = e.components.values.sumByDouble { it.pow(2.0) }
                e.components.forEach { (k,v) ->
                    finalComponents.merge(k, v * weight , ::sum)
                }
            }
        }
        return EmailSparseVector(label = "", components = finalComponents, id = "")
    }

    fun convertResult2(weights: List<Double>, emails: List<EmailSparseVector>): EmailSparseVector {
        val finalComponents = HashMap<String, Double>()
        emails.zip(weights).forEach { (e, weight) ->
            if (weight != 0.0) {
                val eC = e.components.normalize()
                eC.forEach { (k,v) ->
                    finalComponents.merge(k, (v * weight) , ::sum)
                }
            }
        }
        return EmailSparseVector(label = "", components = finalComponents, id = "")
    }

    fun rerunResult(e: EmailSparseVector, stochastic: StochasticComponent) {
        val newEmbedded = trainingComponent.vectors.map { trainingComponent.embed(it, listOf(e)) }
        val newHoldout = trainingComponent.holdout.map { trainingComponent.embed(it, listOf(e)) }

        val newSpam = newEmbedded.filter { it.label == "spam" }
        val newHam = newEmbedded.filter { it.label == "ham" }

        stochastic.holdout = newHoldout
        stochastic.hamVectors = newHam
        stochastic.spamVectors = newSpam
        stochastic.memoizedSpamDist = stochastic.createNormalDist(listOf(1.0), newSpam)
        stochastic.memoizedHamDist = stochastic.createNormalDist(listOf(1.0), newHam)
        val labeler = stochastic.myLabeler(listOf(1.0))
        println("RESULT: ${stochastic.getF1(labeler)}")

    }

    fun doTrain() {
        val results = (0 until trainingComponent.basisCollection.size).take(1).map { index ->
            val embedded = trainingComponent.vectors.map { trainingComponent.embed(it, trainingComponent.basisCollection[index]) }
            val holdout = trainingComponent.holdout.map { trainingComponent.embed(it, trainingComponent.basisCollection[index]) }

            val stochastic = StochasticComponent(trainingComponent.nBases, embedded, holdout)

            val weights = stochastic.doTrain()
            val e = convertResult2(weights, trainingComponent.basisCollection[index])
            rerunResult(e, stochastic)
        }

//        val embedded = trainingComponent.vectors.map { trainingComponent.embed(it, results) }
//        val holdout = trainingComponent.holdout.map { trainingComponent.embed(it, results) }
//
//        val stochastic = StochasticComponent(results.size, embedded, holdout)
//
//        val weights = stochastic.doTrain()
//        val e = convertResult(weights, results)
//
//        rerunResult(e, stochastic)

    }

}

fun main(args: Array<String>) {
    val searcher = SearchUtils.createIndexSearcher("index")
    val predictor = SimpleStochasticTrainer(searcher)
    predictor.doTrain()
}