package edu.unh.cs753.predictors;

import edu.unh.cs.treccar_v2.Data;
import edu.unh.cs753.BayesCounter;
import edu.unh.cs753.utils.SearchUtils;
import org.apache.lucene.search.IndexSearcher;

import java.io.IOException;
import java.util.List;

public class DemoPredictor extends LabelPredictor {

    public DemoPredictor(IndexSearcher s) {
        super(s);

        // Train your method on ham emails (if applicable)
        for(List<String> tokens : retrieveHamEmailTokens()) {
            // Do something with these tokens (they are from ham emails)
        }

        // Train your method on spam emails (if applicable)
        for(List<String> tokens : retrieveSpamEmailTokens()) {
            // Do something with these tokens (they are from spam emails)
        }
    }

    @Override
    public String predict(List<String> tokens) {

        // The predict methods takens tokens from an email, and you must decide if the email is ham or spam using them.
        // Below is a stupid example where you just randomly assign a label.
        // You will want to replace this with your actual classification method
        if (Math.random() > 0.5) {
            return "spam";
        } else {
            return "ham";
        }
    }

    public static void main(String[] args) throws IOException {

        // This is where you test out your predictor by running evaluate
        // Make sure you created "parsed_emails.tsv" and the lucene index! See: DebugHelper
        IndexSearcher searcher = SearchUtils.createIndexSearcher("index");
        LabelPredictor predictor = new DemoPredictor(searcher);
        predictor.evaluate();
    }

}

