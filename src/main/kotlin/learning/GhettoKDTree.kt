package learning

import components.TrainingVectorComponent
import containers.EmailSparseVector
import kernels.SimilarityFuns
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.absoluteValue


class GhettoKDTree(val trainingVectorComponent: TrainingVectorComponent) {

    val cList = getComponentIndexes()
    val memoized = ConcurrentHashMap<EmailSparseVector, List<EmailSparseVector>>()


     fun retrieveCandidate(email: EmailSparseVector): EmailSparseVector {
         val candidates = memoized.computeIfAbsent(email) {
             email.components.flatMap { (component, score) ->
                 retrieveCandidatesByComponent(component.toInt(), score) }
                 .toSet()
                 .toList()
         }

         println(candidates.size)

        return candidates.map { candidate ->
            candidate to SimilarityFuns.simComponentL1Dist(email, candidate) }
            .sortedBy { it.second }
            .first()
            .first
    }

    fun retrieveCandidates(email: EmailSparseVector, weights: List<Double>, k: Int): List<EmailSparseVector> {
        val candidates = email.components.flatMap { (component, score) ->
            retrieveCandidatesByComponent(component.toInt(), score) }
            .toSet()
            .toList()


        return candidates.map { candidate ->
//            candidate to SimilarityFuns.simComponentL1DistWeights(email, candidate, weights) }
        candidate to SimilarityFuns.simComponentL2DistWeights(email, candidate, weights) }
            .sortedBy { it.second }
            .take(k)
            .map { it.first }
    }

    private fun retrieveCandidatesByComponent(index: Int, score: Double): List<EmailSparseVector> {
        val nElements = cList[index].size
        val component = cList[index]

        if (score < component[0].second) {
            return  component.take(20).map { it.first }
        }

        val insertionPoint = component.binarySearch { element ->
            if (element.second < score) -1
            else if (element.second == score) 0
            else 1
        }.absoluteValue

        val b1 = Math.max(0, (insertionPoint - 20))
        val b2 = Math.min(nElements, insertionPoint + 20)
        return component.subList(b1, b2).map { it.first }


//        var highIndex = nElements - 1
//        var lowIndex = 0
//
//        while (lowIndex <= highIndex) {
//            val midPoint = (highIndex + lowIndex) / 2
//            var cur = component[midPoint]
//            if (score < cur.second) {
//                highIndex = midPoint - 1
//            } else if (score > cur.second) {
//                lowIndex = midPoint + 1
//            } else {
//                val b1 = Math.max(0, (midPoint - 1))
//                val b2 = Math.min(nElements, midPoint + 1)
//                return component.subList(b1, b2).map { it.first }
//            }
//        }

//        println("WHAT")
//
//        return emptyList()

    }

    fun getDerivative(index: Int) =
            cList[index]
            .windowed(2, 1, false)
            .map { (v1, v2) ->
//                val dist = SimilarityFuns.simComponentL1Dist(v1.first, v2.first).run { if (this == 0.0) 0.000001 else this  }
                val dist = SimilarityFuns.simComponentL2Dist(v1.first, v2.first)
//                val coordDist = (v1.second - v2.second).absoluteValue.run { if (this == 0.0) 0.000001 else this  }
                val coordDist = (v1.second - v2.second).absoluteValue
                val derivative = if (dist == coordDist) 1.0 else dist / coordDist
                derivative to v1.first.id
//                dist / coordDist
//                dist to coordDist
//                v1.second to v1.first.components.values.sum()
//                v1.second
            }
//                .windowed(3, 1, false)
//                .map { it.sumByDouble { it.first } / it.sumByDouble { it.second } }


    fun getComponentIndexes(): List<ArrayList<Pair<EmailSparseVector, Double>>> {
        val nComponents = trainingVectorComponent.vectors.first().components.size
        val componentLists = (0 until nComponents)
            .toList()
            .map { ArrayList<Pair<EmailSparseVector, Double>>() }


        trainingVectorComponent.vectors.forEach { vec ->
            vec.components.forEach { component, score ->
                componentLists[component.toInt()].add(vec to score)
            }
        }

        componentLists.forEach { cl -> cl.sortBy { it.second } }
        return componentLists
    }

}

fun main(args: Array<String>) {

}