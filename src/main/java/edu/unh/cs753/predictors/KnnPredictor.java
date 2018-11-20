package edu.unh.cs753.predictors;

import edu.unh.cs753.BayesCounter;
import edu.unh.cs753.utils.SearchUtils;
import org.apache.lucene.search.IndexSearcher;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;

public class KnnPredictor extends LabelPredictor {

    private BayesCounter bc = new BayesCounter();
    private double hamCount = 0;
    private double spamCount = 0;
    private HashMap<String, Double> unknownEmailTokens = new HashMap<>();
    Tuple [] distances = new Tuple[10000000];

    public KnnPredictor(IndexSearcher s) {
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
        getUnknownTokens(tokens);

        // 1. Initialize the value of k
        int k = 3;
        int hamScore = 0;
        int spamScore = 0;

        int stop = getDistances(spamDist, hamDist);

        // Sort the list for easy access of top k results
        quickSort(0, stop - 1);

        // Keep track of the classes for the top k elements
        for (int i = 0; i < k; i++) {
            Tuple t = distances[i];
            if (t.label.equals("ham")) {
                hamScore++;
            }
            else {
                spamScore++;
            }
        }

        // Get the most frequent class of the top k rows and return the predicted class.
        if (hamScore > spamScore) {
            return "ham";
        }
        else {
            return "spam";
        }

    }

    /*
     * Custom implementation of the quick sort algorithm to sort the list of Tuples.
     */
    public int Partition(int start, int stop) {

        Tuple t = distances[stop - 1];
        System.out.println(t);
        double pivot = t.score;
        int i = start - 1;

        for (int j = 0; j < pivot; j++) {
            Tuple nt = distances[j];
            if (nt.score <= pivot) {
                i++;
                Tuple temp = distances[i];
                distances[i] = distances[j];
                distances[j] = temp;
            }
        }

        Tuple temp = distances[i + 1];
        distances[i + 1] = distances[distances.length - 1];
        distances[distances.length - 1] = temp;

        return i + 1;
    }

    public void quickSort(int start, int stop) {

        if (start < stop) {

            int partition = Partition(start, stop);

            quickSort(start, stop - 1);
            quickSort(partition, stop);

        }
    }

    /*
     * Calculate the Euclidian distance between each point of training data and each point of test data.
     */
    public int getDistances(HashMap<String, Integer> spamDist, HashMap<String, Integer> hamDist) {

        int index = 0;
        for (String token: unknownEmailTokens.keySet()) {
            if (spamDist.get(token) != null) {
                double dist = Math.log(Math.sqrt(Math.pow(unknownEmailTokens.get(token) - spamDist.get(token), 2)));
                Tuple t = new Tuple(dist, "spam");
                distances[index] = t;
                index++;
            }
            if (hamDist.get(token) != null) {
                double dist = Math.log(Math.sqrt(Math.pow(unknownEmailTokens.get(token) - hamDist.get(token), 2)));
                Tuple t = new Tuple(dist, "ham");
                distances[index] = t;
                index++;
            }
        }
        return index;
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

    /*
     * Custom class to store the distances and their labels.
     */
    private class Tuple {

        double score;
        String label;

        Tuple(double s, String l) {
            this.score = s;
            this.label = l;
        }

    }


    public static void main(String[] args) throws IOException {
        IndexSearcher searcher = SearchUtils.createIndexSearcher("index");
        LabelPredictor predictor = new KnnPredictor(searcher);
        predictor.evaluate();
    }

}

