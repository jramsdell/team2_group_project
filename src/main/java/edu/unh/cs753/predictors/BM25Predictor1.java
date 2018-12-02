package edu.unh.cs753.predictors;

import edu.unh.cs753.utils.SearchUtils;
import org.apache.lucene.document.Document;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;

import java.io.IOException;
import java.util.List;

public class BM25Predictor1 extends LabelPredictor {

    public BM25Predictor1(IndexSearcher s) {
     super(s);


    }

    @Override
    public String predict(List<String> tokens) {

        // The predict methods takens tokens from an email, and you must decide if the email is ham or spam using them.
        // Below is a stupid example where you just randomly assign a label.
        // You will want to replace this with your actual classification method
        List<String> tokenSubset = tokens.subList(0, Math.min(10, tokens.size()));
        Query q = SearchUtils.createStandardBooleanQuery(String.join(" ", tokenSubset), "text");
        int ham = 0;
        int spam = 0;
        try {
            TopDocs topDocs = searcher.search(q, 1);

            for ( ScoreDoc sd : topDocs.scoreDocs) {
                Document doc = searcher.doc(sd.doc);
                String label = doc.get("label");

                if (label.equals("ham")) {
                    ham += 1;
                } else {
                    spam += 1;
                }



            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        if (ham > spam) {
            return "ham";
        } else {
            return "spam";
        }
    }



    public static void main(String[] args) throws IOException {

        // This is where you test out your predictor by running evaluate
        // Make sure you created "parsed_emails.tsv" and the lucene index! See: DebugHelper
        IndexSearcher searcher = SearchUtils.createIndexSearcher("index");
        BM25Predictor1 pre1 = new BM25Predictor1(searcher);
        pre1.evaluate();
    }

}

