package edu.unh.cs753.predictors;

import edu.unh.cs753.BayesCounter;
import edu.unh.cs753.utils.SearchUtils;
import org.apache.lucene.search.IndexSearcher;

import java.io.IOException;
import java.util.List;

public class NaiveBayesPredictor extends LabelPredictor {
    private BayesCounter bc = new BayesCounter();

    public NaiveBayesPredictor(IndexSearcher s) {
        super(s);

        // Train classifier on ham emails
        for(List<String> tokens : retrieveHamEmailTokens()) {
            bc.buildHashMap("ham", tokens);
        }

        // Train classifier on spam emails
        for(List<String> tokens : retrieveSpamEmailTokens()) {
            bc.buildHashMap("spam", tokens);
        }
    }

    @Override
    public String predict(List<String> tokens) {
        return bc.classify(tokens);
    }

    public static void main(String[] args) throws IOException {
        IndexSearcher searcher = SearchUtils.createIndexSearcher("index");
        LabelPredictor predictor = new NaiveBayesPredictor(searcher);
        predictor.evaluate();
    }

}

