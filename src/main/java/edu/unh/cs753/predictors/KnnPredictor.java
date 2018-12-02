package edu.unh.cs753.predictors;

import edu.unh.cs753.BayesCounter;
import edu.unh.cs753.utils.SearchUtils;
import org.apache.lucene.search.IndexSearcher;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class KnnPredictor extends LabelPredictor {

    private BayesCounter bc = new BayesCounter();
    private HashMap<String, Double> unknownEmailTokens = new HashMap<>();
    private Tuple [] distances = new Tuple [10000000];

    public KnnPredictor(IndexSearcher s) {
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
        getUnknownTokens(tokens);
        ArrayList<Tuple> al = new ArrayList<>();

        int k = 3;
        int hamScore = 0;
        int spamScore = 0;

        int num = getDistances(spamDist, hamDist);

        for (int i = 0; i < k; i++) {
            //System.out.println("    i: " + i);
            al.add(getMinimum(num));
        }

        for (int i = 0; i < k; i++) {
            Tuple t = al.get(i);
            if (t.label.equals("spam")) {
                spamScore++;
                //System.out.println("spamScore: " + spamScore);
            }
            else {
                hamScore++;
                //System.out.println("hamScore: " + hamScore);
            }
        }

        if (hamScore > spamScore) {
            return "ham";
        }
        else {
            return "spam";
        }

    }

    /*
     * Utility function to get the lowest Euclidian distance that was calculated
     * and has not yet been added to the list of k nearest neighbors.
     */
    private Tuple getMinimum(int num) {

        Tuple t = distances[0];
        Tuple retTuple = t;
        double min = t.score;
        for (int i = 1; i < num; i++) {
            t = distances[i];
            if (t.score < min && !t.flag) {
                min = t.score;
                //System.out.println(min);
                retTuple = t;
                // Set the flag to true so we don't return the same tuple
                // when the function is called again.
                t.flag = true;
            }
        }
        return retTuple;
    }

    /*
     * Calculate the Euclidian distance between each point of
     * training data and each point of test data.
     */
    private int getDistances(HashMap<String, Integer> spamDist, HashMap<String, Integer> hamDist) {

        int index = 0;
        for (String token: unknownEmailTokens.keySet()) {
            if (spamDist.get(token) != null) {
                double dist = Math.sqrt(Math.pow((unknownEmailTokens.get(token) - spamDist.get(token)), 2.0));
                //System.out.println("spamDist: " + dist);
                Tuple t = new Tuple(dist, "spam", false);
                distances[index] = t;
                index++;
            }
            if (hamDist.get(token) != null) {
                double dist = Math.sqrt(Math.pow((unknownEmailTokens.get(token) - hamDist.get(token)), 2.0));
                //System.out.println("hamDist: " + dist);
                Tuple t = new Tuple(dist, "ham", false);
                distances[index] = t;
                index++;
            }
        }
        return index;
    }

    /*
     * Parse the email tokens and store in a HashMap for
     * easy lookup.
     */
    private void getUnknownTokens(List<String> tokens) {
        for (String token: tokens) {
            if (!unknownEmailTokens.containsKey(token)) {
                unknownEmailTokens.put(token, 0.0);
            }

            double curCount = unknownEmailTokens.get(token);
            unknownEmailTokens.put(token, curCount + 1.0);
        }
    }

    /*
     * Custom class to store the distances and their labels.
     */
    private class Tuple {

        double score;
        String label;
        boolean flag;

        Tuple(double s, String l, boolean f) {
            this.score = s;
            this.label = l;
            this.flag = f;
        }

    }


    public static void main(String[] args) throws IOException {
        IndexSearcher searcher = SearchUtils.createIndexSearcher("index");
        LabelPredictor predictor = new KnnPredictor(searcher);
        predictor.evaluate();
    }

}

