package containers



data class EmailEmbeddedVector(val label: String,
                               val components: List<Double>,
                               val id: String = "")