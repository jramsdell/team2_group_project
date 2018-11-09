package edu.unh.cs753.predictors;

import edu.unh.cs753.utils.SearchUtils;
import org.apache.lucene.document.Document;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;

import java.util.*;

import java.io.IOException;

public class BM25Predictor extends LabelPredictor {

    public BM25Predictor(IndexSearcher s) {
     super(s);


    }

    @Override
    public String predict(List<String> tokens) {

        // The predict methods takens tokens from an email, and you must decide if the email is ham or spam using them.
        // Below is a stupid example where you just randomly assign a label.
        // You will want to replace this with your actual classification method
        List<String> tokenSubset = tokens.subList(0, Math.min(10, tokens.size()));
        Query q = SearchUtils.createStandardBooleanQuery(String.join(" ", tokenSubset), "text");
        try {
            TopDocs topDocs = searcher.search(q, 11);
            int ham = 0;
            int spam = 0;
            for ( ScoreDoc sd : topDocs.scoreDocs) {
                Document doc = searcher.doc(sd.doc);
                String label = doc.get("label");

                if (label.equals("ham")) {
                    ham += 1;
                } else {
                    spam += 1;
                }

                if (ham > spam) {
                    return "ham";
                } else {
                    return "spam";
                }

            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return "ham";
    }



    public static void main(String[] args) throws IOException {

        // This is where you test out your predictor by running evaluate
        // Make sure you created "parsed_emails.tsv" and the lucene index! See: DebugHelper
        IndexSearcher searcher = SearchUtils.createIndexSearcher("index");
        LabelPredictor predictor = new BM25Predictor(searcher);
        predictor.evaluate();
    }

}

