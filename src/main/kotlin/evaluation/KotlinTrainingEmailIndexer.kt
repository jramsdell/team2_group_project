package evaluation

import edu.unh.cs753.utils.IndexUtils
import org.apache.lucene.document.Document
import org.apache.lucene.document.Field
import org.apache.lucene.document.TextField


object KotlinTrainingEmailIndexer {

    fun createIndex(emailTsvLoc: String, indexLoc: String) {
        val emails = KotlinEmailParser.readEmailTsv(emailTsvLoc)
        val (train, _) = KotlinEmailParser.createTestTrainData(emails, 0.5)

        val writer = IndexUtils.createIndexWriter(indexLoc)

        train.forEach { trainingEmail ->
            val doc = Document()
            doc.add(TextField("label", trainingEmail.label, Field.Store.YES))
            doc.add(TextField("id", trainingEmail.emailId, Field.Store.YES))
            doc.add(TextField("text", trainingEmail.tokens.joinToString(" "), Field.Store.YES))
            writer.addDocument(doc)
        }

        writer.commit()
        writer.close()

    }


}

fun main(args: Array<String>) {
    KotlinTrainingEmailIndexer.createIndex("parsed_emails.tsv", "index")
}