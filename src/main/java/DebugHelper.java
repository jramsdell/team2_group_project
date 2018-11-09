import evaluation.KotlinEmailParser;
import evaluation.KotlinTrainingEmailIndexer;

import java.io.IOException;

public class DebugHelper {

    /**
     * Step 1: parse emails in the Spam Directory
     */
    public static void parseEmails() {

        // NOTE: you must change spamEmailDirectory to point to the spam directory on your laptop
        // Make SURE that the spam directory is unzipped

        String spamEmailDirectory = "/home/rachel/ir/project/trec07p"; // replace this path!

        KotlinEmailParser.INSTANCE.parseEmails(spamEmailDirectory);
    }

    /**
     * Step 2: create a lucene index out of the training emails
     */

    public static void createTrainingIndex() {
        KotlinTrainingEmailIndexer.INSTANCE.createIndex("parsed_emails.tsv", "index");
    }


    public static void main(String[] args) throws IOException {

        // This parses the spam directory for emails and create "parsed_emails.tsv" in the project directory:
        parseEmails();

        // This will create a lucene index out of the training emails (you must run parseEmails() first to create  parsed_emails.tsv)
        createTrainingIndex();
    }

}
