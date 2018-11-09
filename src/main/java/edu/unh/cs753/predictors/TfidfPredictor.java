package edu.unh.cs753.predictors;

import edu.unh.cs753.BayesCounter;
import edu.unh.cs753.utils.SearchUtils;
import org.apache.lucene.search.IndexSearcher;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;

public class TfidfPredictor extends LabelPredictor {
    private BayesCounter bc = new BayesCounter();

    public TfidfPredictor(IndexSearcher s) {
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
        HashMap<String, Integer> spamDist = bc.bayesMap.get("spam");
        HashMap<String, Integer> hamDist = bc.bayesMap.get("ham");

        double hamScore = 0;
        double spamScore = 0;
        double N = getNumberOfDocsInCorpus();

        for (String token : tokens) {

            double df = getDocFrequency(token);
            spamScore += Math.log((spamDist.getOrDefault(token, 1)) * Math.log(N/df));
            hamScore += Math.log((hamDist.getOrDefault(token, 1)) * Math.log(N/df));

        }

        if (hamScore > spamScore) {
            return "ham";
        }
        else {
            return "spam";
        }
    }

    public static void main(String[] args) throws IOException {
        IndexSearcher searcher = SearchUtils.createIndexSearcher("index");
        LabelPredictor predictor = new TfidfPredictor(searcher);
        predictor.evaluate();
    }

}

