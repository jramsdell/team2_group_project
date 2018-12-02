package components

import com.google.common.util.concurrent.AtomicDouble
import containers.EmailEmbeddedVector
import containers.EmailSparseVector
import kernels.*
import org.apache.lucene.index.Term
import org.apache.lucene.search.IndexSearcher
import org.apache.lucene.search.TermQuery
import utils.*
import java.lang.Double.sum
import java.util.*
import kotlin.collections.ArrayList


class TrainingVectorComponent(val searcher: IndexSearcher, val nBases: Int = 80, val nSets: Int = 5, var nElements: Int = 3000) {
    val vectors = ArrayList<EmailSparseVector>()
//    val hamMatrix = (0 until 50).map { ArrayList<Double>() }
//    val spamMatrix = (0 until 50).map { ArrayList<Double>() }

//    val holdoutHamMatrix = (0 until 50).map { ArrayList<Double>() }
//    val holdoutSpamMatrix = (0 until 50).map { ArrayList<Double>() }


    val basisVectors = ArrayList<EmailSparseVector>()
    val basisCollection = ArrayList<List<EmailSparseVector>>()
    val holdout = ArrayList<EmailSparseVector>()
    val extras = ArrayList<EmailSparseVector>()
//    val coMap = HashMap<String, HashMap<String, Double>>()

    val kernel = SoftplusKernel(SimilarityFuns::simComponentCosine)
    val kernel2 = SoftplusKernel(SimilarityFuns::simBigramCosine)
//    val kernel3 = SoftplusKernel({ e1, e2 -> SimilarityFuns.simComponentDotCovariance(e1, e2, coMap)})
    val kernel4 = SoftplusKernel(SimilarityFuns::simComponentDotKld)
    val kernel5 = SoftplusKernel(SimilarityFuns::simOverlap)
    val kernel6 = SoftplusKernel(SimilarityFuns::simBigramOverlap)
    val kernel7 = SoftplusKernel(SimilarityFuns::simComponentDotKldBigram)



    init {
        train()
        basisVectors.clear()
//        getCovariance()
    }

//    fun doExpand() {
//
//        coMap.entries.forEach { (k,v) ->
//            val newMap = HashMap<String, Double>()
//            v.entries.forEach { (k2, v2) ->
//                coMap[k2]!!.entries.forEach { (k3, v3) ->
//                    newMap.merge(k3, v3 * v2, ::sum)
//                }
//                newMap.merge(k2, v2, ::sum)
//            }
//
//            val total = newMap.values.sum()
//            v.clear()
//            newMap.forEach { (k2, v2) ->
//                v[k2] = v2 / total
//            }
//
//        }
//
//    }

//    fun getCovariance() {
//        vectors.forEach { vector ->
//            val mostFreq = vector.components.takeMostFrequent(5)
//            mostFreq.forEach { (c1, d1) ->
//                mostFreq.forEach { (c2, d2) ->
//                    if (c1 !in coMap)
//                        coMap[c1] = HashMap()
//                    if (c2 !in coMap)
//                        coMap[c2] = HashMap()
//
//                    coMap[c1]!!.merge(c2, d1 * d2, ::sum)
////                    coMap[c2]!!.merge(c1, d1 * d2, ::sum)
//                }
//            }
//        }

//        val total = coMap.entries.sumByDouble { it.value.entries.sumByDouble { it.value } }
//        val instances = HashMap<String, Double>()
//
//        coMap.entries.forEach { (k,v) ->
//            v.entries.forEach { (k2, v2) ->
//                instances.merge(k2, v2, ::sum)
//            }
//        }


//        val final = instances
//
//        coMap.entries.forEach { (k,v) ->
//            val total = v.values.sum()
//            v.forEach { (k2, v2) -> v[k2] = v2 / total }
//        }
//    }


    private fun train() {
        val nDocs = searcher.indexReader.numDocs()
        var randomDocs = (0 until nDocs).shuffled(Random(21)).take(nElements)
//            .map(this::extractEmail)
            .pmap { extractEmail(it) }
            .filter { it.components.size > 0}

        nElements = randomDocs.size

        (0 until nSets).forEach {
            randomDocs
                .take(nBases)
                .toList()
                .apply { basisCollection.add(this) }

            randomDocs = randomDocs.drop(nBases)
        }

        randomDocs.take(500)
            .mapTo(extras) { it }

        randomDocs = randomDocs.drop(500)

//        randomDocs
//            .take(nBases)
//            .mapTo(basisVectors) { it }

        val step = randomDocs.size / 2


//        val split = randomDocs.drop(nBases)
//            .pmap { embed(it) }

        randomDocs.take(step)
            .mapTo(vectors) {it}
        randomDocs.drop(step)
            .mapTo(holdout) {it}

    }





    private fun extractEmail(docId: Int): EmailSparseVector  {
        val doc = searcher.doc(docId)
        val label = doc.get("label")
        val id = doc.get("id")

        // Create frequency dist of tokens
        val dist = doc.get("text")
            .split(" ")
//            .flatMap { createCharacterGrams(it, 4) }
//            .run { createBigrams(this) }
            .groupingBy { it }
            .eachCount()
            .map { it.key to it.value.toDouble() }
//            .map { it.key to Math.log(it.value.toDouble()) + 1.0 }
            .toMap()

        val dist2 = doc.get("text")
            .split(" ")
//            .flatMap { createCharacterGrams(it, 4) }
            .run { createBigrams(this) }
//            .flatMap { createCharacterGrams(it, 3) }
            .groupingBy { it }
            .eachCount()
            .map { it.key to it.value.toDouble() }
//            .map { it.key to Math.log(it.value.toDouble()) + 1.0 }
            .toMap()

        return EmailSparseVector(label = label, components = dist, id = id, bigrams = dist2)
    }




    fun createCharacterGrams(token: String, n: Int) =
        token.windowed(n, 1, false)

    fun createBigrams(tokens: List<String>): List<String> =
            tokens
                .windowed(2, 1, false)
                .map { it[0] + it[1]  }



    fun embed(v: EmailSparseVector, bVectors: List<EmailSparseVector>): EmailSparseVector {
//        val transformedComponents = basisCollection[basisIndex].mapIndexed { index, basis ->
            val transformedComponents = bVectors.flatMap { basis ->
//            val key = index.toString()
                val results = listOf(
                        kernel.sim(v.components, basis.components)
//                        kernel2.sim(v.bigrams, basis.bigrams)
//                        kernel3.sim(v.components, basis.components),
//                        kernel4.sim(v.components, basis.components),
//                        kernel5.sim(v.components, basis.components),
//                        kernel6.sim(v.bigrams, basis.bigrams),
//                        kernel7.sim(v.bigrams, basis.bigrams)
                ).map { if (it.isNaN()) 0.0 else it }
        results }
                .mapIndexed { index, d -> index.toString() to d  }
                .toMap()

        return EmailSparseVector(label = v.label, components = transformedComponents, id = v.id)
    }



}