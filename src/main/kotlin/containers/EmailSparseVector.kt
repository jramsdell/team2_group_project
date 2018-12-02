package containers

import representations.KernelType
import representations.Representation
import representations.RepresentationType


data class EmailSparseVector(val label: String, val components: Map<String, Double>, val id: String = "", val bigrams: Map<String, Double> = emptyMap(),
                             val representations: Map<Pair<RepresentationType, KernelType>, Representation> = emptyMap(), var score: Double = 0.0)