package edu.unh.cs753.predictors;

import edu.unh.cs753.BayesCounter;
import edu.unh.cs753.utils.SearchUtils;
import org.apache.lucene.search.IndexSearcher;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;

public class CosinePredictor extends LabelPredictor {

    private BayesCounter bc = new BayesCounter();
    private double hamCount = 0;
    private double spamCount = 0;
    private HashMap<String, Double> spamCentroid  = new HashMap<>();
    private HashMap<String, Double> hamCentroid = new HashMap<>();
    private HashMap<String, Double> unknownEmailTokens = new HashMap<>();

    public CosinePredictor(IndexSearcher s) {
        super(s);

        // Train classifier on ham emails
        for(List<String> tokens : retrieveHamEmailTokens()) {
            bc.buildHashMap("ham", tokens);
            hamCount += 1;
        }

        // Train classifier on spam emails
        for(List<String> tokens : retrieveSpamEmailTokens()) {
            bc.buildHashMap("spam", tokens);
            spamCount += 1;
        }
    }

    @Override
    public String predict(List<String> tokens) {

        HashMap<String, Integer> spamDist = bc.bayesMap.get("spam");
        HashMap<String, Integer> hamDist = bc.bayesMap.get("ham");
       // System.out.println("before centroid computation");
        computeCentroids(spamDist, hamDist);
        //System.out.println("after centroid computation");
        getUnknownTokens(tokens);
        //System.out.println("unknown tokens retrieved");

        double hamScore = 0;
        double spamScore = 0;
        double hamDotProd = 0;
        double spamDotProd = 0;
        double hamCosine = 0;
        double spamCosine = 0;
        double unknownCosine = 0;

        // Get the cosine similarity between the spam and ham centroids and
        // an unknown email.
        for (String token : tokens) {
            // 1. Get the corresponding token in the centroids and multiply with the current token.
            // Keep a running sum to compute the dot product.
            Double knownSpam = spamCentroid.getOrDefault(token, 1.0);
            Double knownHam = hamCentroid.getOrDefault(token, 1.0);
            Double unknown = unknownEmailTokens.getOrDefault(token, 1.0);

            hamDotProd += knownSpam * unknown;
            spamDotProd += knownHam * unknown;

            // 2. Keep a running sum of all of the squares of all the term frequencies in the centroid.
            hamCosine += (knownHam*knownHam);
            spamCosine += (knownSpam*knownSpam);
            unknownCosine += (unknown*unknown);
        }

        // Square root the total of the term frequencies from step 2.
        hamCosine = Math.sqrt(hamCosine);
        spamCosine = Math.sqrt(spamCosine);
        unknownCosine = Math.sqrt(unknownCosine);

        // Divide first computation by second.
        hamScore = hamDotProd/(hamCosine * unknownCosine);
        spamScore = spamDotProd/(spamCosine * unknownCosine);

        if (hamScore > spamScore) {
            return "ham";
        }
        else {
            return "spam";
        }

    }

    public void getUnknownTokens(List<String> tokens) {
        for (String token: tokens) {
            if (!unknownEmailTokens.containsKey(token)) {
                unknownEmailTokens.put(token, 0.0);
            }

            double curCount = unknownEmailTokens.get(token);
            unknownEmailTokens.put(token, curCount + 1);
        }
    }

    public void computeCentroids(HashMap<String, Integer>spamDist, HashMap<String, Integer>hamDist) {

        // For each email token divide by the total number of spam/ham documents.
        for(String key: spamDist.keySet()) {
            double value = spamDist.get(key);
            spamCentroid.put(key, value/spamCount);
        }

        for(String key: hamDist.keySet()) {
            double value = hamDist.get(key);
            hamCentroid.put(key, value/hamCount);
        }
    }

    public static void main(String[] args) throws IOException {
        IndexSearcher searcher = SearchUtils.createIndexSearcher("index");
        LabelPredictor predictor = new CosinePredictor(searcher);
        predictor.evaluate();
    }

}

