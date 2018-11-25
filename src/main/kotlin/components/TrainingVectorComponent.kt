package components

import containers.EmailSparseVector
import kernels.GaussianKernel
import kernels.LaplacianKernel
import kernels.SimilarityFuns
import org.apache.lucene.search.IndexSearcher
import utils.normalize
import java.util.*
import kotlin.collections.ArrayList


class TrainingVectorComponent(val searcher: IndexSearcher) {
    val vectors = ArrayList<EmailSparseVector>()
    val basisVectors = ArrayList<EmailSparseVector>()
    val kernel = LaplacianKernel(SimilarityFuns::simComponentCosine)
    val kernel2 = LaplacianKernel(SimilarityFuns::simComponentCosine)
    val holdout = ArrayList<EmailSparseVector>()

    init {
        train()
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

    fun getF1(caller: (EmailSparseVector) -> String): Double {
        var tp = 0.0
        var tn = 0.0
        var fn = 0.0
        var fp = 0.0

        vectors.forEach { v ->
            val called = caller(v)
            if (v.label == "spam" && called == "spam") { tp += 1.0 }
            else if (v.label == "spam" && called == "ham") { fn += 1.0 }
            else if (v.label == "ham" && called == "ham") { tn += 1.0 }
            else { fp += 1.0 }
        }

        val precision = tp.toDouble() / (tp + fp)
        val recall = tp.toDouble() / (tp + fn)
        val f1 = (2 * (precision * recall) / (precision + recall)).run { if(isNaN()) 0.0 else this }
        val precision2 = tn.toDouble() / (tn + fn)
        val recall2 = tn.toDouble() / (tn + fp)
        val f2 = (2 * (precision2 * recall2) / (precision2 + recall2)).run { if(isNaN()) 0.0 else this }

        return (f1 + f2) / 2.0
    }


    private fun train() {
        val nDocs = searcher.indexReader.numDocs()
//        val randomDocs = (0 until nDocs).shuffled(Random(21)).take(20)
        var nElements = 1000
        val nBases = 80
        val randomDocs = (0 until nDocs).shuffled(Random(21)).take(nElements)
            .map(this::extractEmail)
            .filter { it.components.size > 0}

        nElements = randomDocs.size

        randomDocs
            .take(nBases)
            .mapTo(basisVectors) { it }

        val split = randomDocs.drop(nBases)
            .map { embed(it) }




        split.take((nElements - nBases) / 2)
            .mapTo(vectors) {it}
        split.drop((nElements - nBases) / 2)
            .mapTo(holdout) {it}

    }

    private fun extractEmail(docId: Int): EmailSparseVector  {
        val doc = searcher.doc(docId)
        val label = doc.get("label")
        val id = doc.get("id")

        // Create frequency dist of tokens
        val dist = doc.get("text")
            .split(" ")
            .flatMap { createTriCharGrams(it) }
            .groupingBy { it }
            .eachCount()
            .map { it.key to it.value.toDouble() }
            .toMap()

        return EmailSparseVector(label = label, components = dist, id = id)
    }

    fun createTriCharGrams(token: String) =
        token.windowed(3, 1, false)


    fun embed(v: EmailSparseVector): EmailSparseVector {
        val transformedComponents = basisVectors.mapIndexed { index, basis ->
            val key = index.toString()
//            val result = SimilarityFuns.simOverlap(v, basis)
//            val result = SimilarityFuns.simComponentL1Dist(basis, v)
//            val result = kernel.sim(v, basis)
            val result = kernel2.sim(v, basis)
//            val result = SimilarityFuns.simComponentCosine(v, basis)
//            println(result)
            key to result
        }.toMap()

        return EmailSparseVector(label = v.label, components = transformedComponents, id = v.id)
    }

    fun embed2(v: EmailSparseVector): EmailSparseVector {
        val transformedComponents = basisVectors.mapIndexed { index, basis ->
            val key = index.toString()
            val result = SimilarityFuns.simComponentL2Dist(v, basis)
            key to result
        }.toMap()

        return EmailSparseVector(label = v.label, components = transformedComponents, id = v.id)
    }

}