package edu.unh.cs753;

import java.util.HashMap;
import java.util.List;

public class BayesCounter {

    /*
     * Makes a new BayesCounter with empty hash map.
     */
    private final HashMap<String, HashMap<String, Integer>> bayesMap;

    public BayesCounter() {
        bayesMap = new HashMap<>();
    }

    /*
     * Parse the email tokens and make a hash map of hash maps for the class passed as a parameter.
     * Example state of the map after this function: {Spam => ["Viagra", 1000], ["great", 987]}
     *
     */
    public void buildHashMap(String docClass, List<String> emailTokens) {

        // Verify that the first parameter is valid
        if (!(docClass.equals("ham")) && !(docClass.equals("spam"))) {
            System.out.print("Error: Invalid class type. \n Options: 'spam, 'ham' ");
            return;
        }

        // Initialize the outer map for the document class.
        HashMap<String, Integer> classMap = new HashMap();
        bayesMap.put(docClass, classMap);

        for (String token : emailTokens) {

            int curCount = classMap.get(token);
            classMap.put(token, curCount + 1);

        }

    }

    /*
     * Parse the tokens of the email passed as a parameter and sum the counts of each word.
     * Return the document class with the larger count.
     *
     */
    public String classify(List<String> tokens) {

        HashMap<String, Integer> spamDist = bayesMap.get("spam");
        HashMap<String, Integer> hamDist = bayesMap.get("ham");

        double hamScore = 0;
        double spamScore = 0;

        for (String token : tokens) {
            spamScore += Math.log(spamDist.get(token));
            hamScore += Math.log(hamDist.get(token));
        }

        if (hamScore > spamScore) {
            return "Ham";
        }
        else {
            return "Spam";
        }
    }

}
