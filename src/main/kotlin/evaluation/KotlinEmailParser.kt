package evaluation

import utils.KotlinDataUtils
import java.io.File
import java.util.*


object KotlinEmailParser {

    public fun parseEmails(spamDirLoc: String) {
        val dataLoc = spamDirLoc + "/data/"
        val baseLabels = KotlinDataUtils.retrieveLabelsFromIndex(spamDirLoc).entries.map { it.toPair() }.toMap()
        var counter = 0
        val validEmails = baseLabels.mapNotNull { (emailId, label) ->
            counter += 1
            if (counter % 1000 == 0) println(counter)
            KotlinEmail.readEmail(dataLoc + emailId, label) }
            .map { it.emailId to it }
            .toMap()


        val out = File("parsed_emails.tsv").bufferedWriter()

        validEmails.forEach { email ->
            with (email.value) {
                out.write("$emailId\t$label\t${tokens.joinToString(" ")}\n")
            }
        }

        out.close()
    }

    public fun readEmailTsv(emailTsvLoc: String) =
        File(emailTsvLoc).bufferedReader()
            .readLines()
            .map { line ->
                val (emailId, label, tokenString) = line.split("\t")
                val tokens = tokenString.split(" ")
                KotlinEmail(
                        emailId = emailId,
                        label = label,
                        subject = "",
                        text = "",
                        tokens = tokens )

            }

    public fun createTestTrainData(emails: List<KotlinEmail>, trainPercent: Double): Pair<List<KotlinEmail>, List<KotlinEmail>> {
        val nTrain = (emails.size * trainPercent).toInt()
        val shuffled = emails.shuffled(Random(12384))
        val train = shuffled.subList(0, nTrain)
        val test = shuffled.subList(nTrain, emails.size)
        return train to test
    }
}


fun main(args: Array<String>) {
    val spamDir = "/home/hcgs/data_science/data/spam/trec07p/"
//    KotlinEmailParser().parseEmails(spamDir)
}