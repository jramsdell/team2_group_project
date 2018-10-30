package utils

import java.io.File


object KotlinDataUtils {

    fun retrieveLabelsFromIndex(spamDataDirectory: String): Map<String, String> {
        val indexDir = spamDataDirectory + "/full/index"
        val indexFile = File(indexDir).bufferedReader()

        return indexFile
            .readLines()
            .map { line -> line.split(" ").let { it[0] to it[1].split("/").last() } }
            .toMap()
    }

    fun doSplit(labels: Map<String, String>, trainPortion: Double): Pair<Map<String, String>, Map<String, String>> {
        val randomizedData = labels.entries.map { it.toPair() }.shuffled()
        val nTrain = (trainPortion * randomizedData.size).toInt()
        val train = randomizedData.subList(0, nTrain)
        val test = randomizedData.subList(nTrain, randomizedData.size)

        return train.toMap() to test.toMap()
    }
}