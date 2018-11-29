package components

import com.google.common.util.concurrent.AtomicDouble
import containers.EmailEmbeddedVector
import containers.EmailSparseVector
import kernels.*
import org.apache.lucene.search.IndexSearcher
import utils.*
import java.lang.Double.sum
import java.util.*
import kotlin.collections.ArrayList


class TrainingVectorComponent(val searcher: IndexSearcher) {
    var nBases = 30
    var nSets = 5
    val vectors = ArrayList<EmailSparseVector>()
//    val hamMatrix = (0 until 50).map { ArrayList<Double>() }
//    val spamMatrix = (0 until 50).map { ArrayList<Double>() }

//    val holdoutHamMatrix = (0 until 50).map { ArrayList<Double>() }
//    val holdoutSpamMatrix = (0 until 50).map { ArrayList<Double>() }


    val basisVectors = ArrayList<EmailSparseVector>()
    val basisCollection = ArrayList<List<EmailSparseVector>>()
    val holdout = ArrayList<EmailSparseVector>()
    val extras = ArrayList<EmailSparseVector>()
    val coMap = HashMap<String, HashMap<String, Double>>()
//    val kernel = SoftplusKernel({ e1, e2 -> SimilarityFuns.simComponentDotCovariance(e1, e2, coMap)})
    val kernel = SoftplusKernel(SimilarityFuns::simBigramCosine)
    val kernel2 = SoftplusKernel(SimilarityFuns::simComponentCosine)
    val kernel3 = SoftplusKernel(SimilarityFuns::simComponentDot)
    val kernel4 = SoftplusKernel(SimilarityFuns::simBigramDot)
    val kernel5 = SoftplusKernel(SimilarityFuns::simOverlap)
    val kernel6 = SoftplusKernel(SimilarityFuns::simBigramOverlap)

    init {
        train()
        basisVectors.clear()
        getCovariance()
    }

    fun getCovariance() {

        vectors.forEach { vector ->
            val mostFreq = vector.components.takeMostFrequent(5)
            mostFreq.forEach { c1, d1 ->
                mostFreq.forEach { c2, d2 ->
                    if (c1 !in coMap)
                        coMap[c1] = HashMap()

                    coMap[c1]!!.merge(c2, d1 * d2, ::sum)
                }
                val total = coMap[c1]!!.values.sum()
                coMap[c1]!!.forEach { (k,v) -> coMap[c1]!![k] = v / total }
//                coMap[c1]!! = coMap[c1]!!.normalize()
            }
        }

    }

    fun findOrthogonal(): ArrayList<EmailSparseVector> {
        val nDocs = searcher.indexReader.numDocs()
        var randomDocs = (0 until nDocs).shuffled(Random(21)).take(5000)
            .map(this::extractEmail)
            .filter { it.components.values.sum() > 200 }

        val bases = ArrayList<EmailSparseVector>()
        bases.add(randomDocs.first())

        randomDocs = randomDocs.drop(1)

        while (bases.size < 10) {
            val nextBase = randomDocs.withIndex().minBy { (index, doc) ->
                bases.map { base ->
                    SimilarityFuns.simOverlap(doc, base) }
                    .max()!!
            }!!

            bases.add(nextBase.value)
            randomDocs = randomDocs.filterIndexed { index, emailSparseVector -> index != nextBase.index  }
        }

        return bases
    }

    private fun train() {
        val nDocs = searcher.indexReader.numDocs()
        var nElements = 2000
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
            .flatMap { createCharacterGrams(it, 4) }
//            .run { createBigrams(this) }
            .groupingBy { it }
            .eachCount()
            .map { it.key to it.value.toDouble() }
//            .map { it.key to Math.log(it.value.toDouble()) + 1.0 }
            .toMap()

        val dist2 = doc.get("text")
            .split(" ")
//            .flatMap { createCharacterGrams(it, 3) }
//            .run { createBigrams(this) }
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
                        kernel.sim(v, basis),
                        kernel2.sim(v, basis),
                        kernel3.sim(v, basis),
                        kernel4.sim(v, basis),
                        kernel5.sim(v, basis),
                        kernel6.sim(v, basis)
                )
        results }
                .mapIndexed { index, d -> index.toString() to d  }
                .toMap()

        return EmailSparseVector(label = v.label, components = transformedComponents, id = v.id)
    }



}