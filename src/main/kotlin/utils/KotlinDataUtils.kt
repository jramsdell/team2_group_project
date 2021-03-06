package utils

import java.io.File
import java.util.*


public object KotlinDataUtils {

    public fun retrieveLabelsFromIndex(spamDataDirectory: String): Map<String, String> {
        val indexDir = spamDataDirectory + "/full/index"
        val indexFile = File(indexDir).bufferedReader()

        val indices = indexFile
            .readLines()
            .map { line -> line.split(" ").let { it[1].split("/").last() to it[0] } }
            .toMap()

        indexFile.close()
        return indices
    }

    public fun doSplit(labels: Map<String, String>, trainPortion: Double): Pair<Map<String, String>, Map<String, String>> {
        val randomizedData = labels.entries.map { it.toPair() }.shuffled(Random(21293813))
        val nTrain = (trainPortion * randomizedData.size).toInt()
        val train = randomizedData.subList(0, nTrain)
        val test = randomizedData.subList(nTrain, randomizedData.size)

        return train.toMap() to test.toMap()
    }
}