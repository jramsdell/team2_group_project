package learning.training

import components.StochasticComponent
import components.TrainingVectorComponent
import containers.EmailSparseVector
import edu.unh.cs753.utils.SearchUtils
import learning.GhettoKDTree
import org.apache.lucene.search.IndexSearcher
import utils.*
import java.lang.Double.sum
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.ln
import kotlin.math.pow


class SimpleStochasticTrainer(val searcher: IndexSearcher) {
    val trainingComponent = TrainingVectorComponent(searcher)

    fun convertResult(weights: List<Double>, emails: List<EmailSparseVector>): EmailSparseVector {
        val finalComponents = ConcurrentHashMap<String, Double>()
        val wNorm = weights.sumByDouble { it.pow(2.0) }
        emails.zip(weights).forEachParallel { (e, weight) ->
            if (weight != 0.0) {
                val eC = e.components.normalize()
                val vNorm = eC.values.sumByDouble { it.pow(2.0) }
                eC.forEach { (k,v) ->
                    finalComponents.merge(k,  (v / vNorm) * weight , ::sum)
                }
            }
        }
        return EmailSparseVector(label = "", components = finalComponents, id = "")
    }

    fun convertResult2(weights: List<Double>, emails: List<EmailSparseVector>): EmailSparseVector {
        val finalComponents = ConcurrentHashMap<String, Double>()
        emails.zip(weights).forEachParallel { (e, weight) ->
            if (weight != 0.0) {
                val eC = e.components.normalize()
                eC.forEach { (k,v) ->
                    finalComponents.merge(k, (v * weight) , ::sum)
                }
            }
        }
        return EmailSparseVector(label = "", components = finalComponents, id = "")
    }

    fun convertResult3(weights: List<Double>, emails: List<EmailSparseVector>): EmailSparseVector {
        val nBasis = weights.size / emails.size

        val finalComponents = ConcurrentHashMap<String, Double>()
        val finalBigrams = ConcurrentHashMap<String, Double>()


        weights.chunked(nBasis)
            .forEachIndexed { index, chunk ->
                val e = emails[index]
                val eC = e.components.normalize()
                val eC2 = e.bigrams.normalize()

                val v1 = chunk[0]
//                val v2 = chunk[1]

                if (v1 != 0.0) {
                    eC.forEach { (k,v) ->
                        finalComponents.merge(k, (v * (v1)) , ::sum)
                    }
                }

//                if (v2 != 0.0) {
//                    eC2.forEach { (k,v) ->
//                        finalBigrams.merge(k, (v * v1) , ::sum)
//                    }
//                }


            }



        return EmailSparseVector(label = "", components = finalComponents, id = "", bigrams = finalBigrams)

    }


    fun rerunResult(e: EmailSparseVector, stochastic: StochasticComponent) {
        val newEmbedded = trainingComponent.vectors.map { trainingComponent.embed(it, listOf(e)) }
        val newHoldout = trainingComponent.holdout.map { trainingComponent.embed(it, listOf(e)) }

        val newSpam = newEmbedded.filter { it.label == "spam" }
        val newHam = newEmbedded.filter { it.label == "ham" }

        val w = listOf(1.0, 1.0)

        stochastic.holdout = newHoldout
        stochastic.hamVectors = newHam
        stochastic.spamVectors = newSpam
        stochastic.memoizedSpamDist = stochastic.createNormalDist(w, newSpam)
        stochastic.memoizedHamDist = stochastic.createNormalDist(w, newHam)
        val labeler = stochastic.myLabeler(w)
        println("RESULT: ${stochastic.getF1(labeler)}")

    }

    fun nextLayer(curLayer: List<EmailSparseVector>, nChunks: Int) = curLayer
    .shuffled()
    .apply { println("====NEXT====\n\n") }
    .chunked(nChunks)
    .flatMap { emails ->
        val embedded = trainingComponent.vectors.map { trainingComponent.embed(it, emails) }
        val holdout = trainingComponent.holdout.map { trainingComponent.embed(it, emails) }

        val stochastic = StochasticComponent(emails.size, embedded, holdout)
        val weights = stochastic.doTrain()
        emails.filterIndexed { index, e -> weights[index] != 0.0  }

    }


    fun doTrain() {
        val results = (0 until trainingComponent.basisCollection.size).map { index ->
                val embedded = trainingComponent.vectors.map { trainingComponent.embed(it, trainingComponent.basisCollection[index]) }
                val holdout = trainingComponent.holdout.map { trainingComponent.embed(it, trainingComponent.basisCollection[index]) }

                val stochastic = StochasticComponent(embedded.first().components.size, embedded, holdout)

                val weights = stochastic.doTrain(true, 1800)
//            val e = convertResult2(weights, trainingComponent.basisCollection[index])
//            rerunResult(e, stochastic)

//            val e = convertResult2(weights, trainingComponent.basisCollection[index])
//            e
        }
//            .run { nextLayer(this, ) }
//            .run { nextLayer(this, 70) }
//            .run { nextLayer(this, 100) }

//        val embedded = (trainingComponent.vectors + trainingComponent.extras).map { trainingComponent.embed(it, results) }
//        val holdout = trainingComponent.holdout.map { trainingComponent.embed(it, results) }
//        val stochastic = StochasticComponent(results.size, embedded, holdout)
//        val weights = stochastic.doTrain2(winnow = false, nIterations = 8)

//        val e = convertResult(weights, results)
//        rerunResult(e, stochastic)


    }

}

fun main(args: Array<String>) {
    val searcher = SearchUtils.createIndexSearcher("index")
    val predictor = SimpleStochasticTrainer(searcher)
    predictor.doTrain()
}