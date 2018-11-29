package containers


data class EmailSparseVector(val label: String, val components: Map<String, Double>, val id: String = "", val bigrams: Map<String, Double> = emptyMap())