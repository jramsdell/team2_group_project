package edu.unh.cs753.predictors;

import edu.unh.cs753.BayesCounter;
import edu.unh.cs753.utils.SearchUtils;
import org.apache.lucene.search.IndexSearcher;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class NaiveBayesQuadgramPredictor extends LabelPredictor {
    private BayesCounter bc = new BayesCounter();

    public NaiveBayesQuadgramPredictor(IndexSearcher s) {
        super(s);

        // Train classifier on ham emails
        for(List<String> tokens : retrieveHamEmailTokens()) {
            bc.buildQuadgramsHashMap("ham", tokens);
        }

        // Train classifier on spam emails
        for(List<String> tokens : retrieveSpamEmailTokens()) {
            bc.buildQuadgramsHashMap("spam", tokens);
        }
    }

    @Override
    public String predict(List<String> tokens) {
        return bc.classifyWithQuadgrams(tokens);
    }

    public ArrayList<Double> getScores(List<String> tokens) {
        return bc.getQuadramScores(tokens);
    }

    public static void main(String[] args) throws IOException {
        IndexSearcher searcher = SearchUtils.createIndexSearcher("index");
        LabelPredictor predictor = new NaiveBayesQuadgramPredictor(searcher);
        predictor.evaluate();
    }

}

