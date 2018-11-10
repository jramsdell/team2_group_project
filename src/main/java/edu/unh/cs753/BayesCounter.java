package edu.unh.cs753;

import java.util.HashMap;
import java.util.List;

public class BayesCounter {

    /*
     * Makes a new BayesCounter with empty hash map.
     */
    public final HashMap<String, HashMap<String, Integer>> bayesMap;

    public BayesCounter() {
        bayesMap = new HashMap<>();
    }

    /*
     * Parse the email tokens and make a hash map of hash maps for the class passed as a parameter.
     * Example state of the map after this function: {Spam => ["Viagra", 1000], ["great", 987]}
     * This function is for training and should therefore be called on the training set of emails.
     */
    public void buildHashMap(String docClass, List<String> emailTokens) {

        // Verify that the first parameter is valid
        if (!(docClass.equals("ham")) && !(docClass.equals("spam"))) {
            System.out.print("Error: Invalid class type. \n Options: 'spam, 'ham' ");
            return;
        }

        // Initialize the outer map for the document class.
        // If a hash map has not yet been initialized for this class, create one.
        // Otherwise, just grab the one that already exists.
        if (bayesMap.get(docClass) == null) {
            HashMap<String, Integer> classMap = new HashMap();
            bayesMap.put(docClass, classMap);
        }

        HashMap<String, Integer> curMap = bayesMap.get(docClass);

        for (String token : emailTokens) {
            if (!curMap.containsKey(token)) {
                curMap.put(token, 0);
            }

            int curCount = curMap.get(token);
            curMap.put(token, curCount + 1);

        }

    }

    /*
     * Parse the tokens of the email passed as a parameter and sum the counts of each word.
     * Return the document class with the larger count.
     * This function is for evaluation and should therefore be called on the evaluation set of emails.
     */
    public String classify(List<String> tokens) {

        HashMap<String, Integer> spamDist = bayesMap.get("spam");
        HashMap<String, Integer> hamDist = bayesMap.get("ham");

        double hamScore = 0;
        double spamScore = 0;

        for (String token : tokens) {
            spamScore += Math.log(spamDist.getOrDefault(token, 1));
            hamScore += Math.log(hamDist.getOrDefault(token, 1));
        }

        if (hamScore > spamScore) {
            return "ham";
        }
        else {
            return "spam";
        }
    }

    /*
     * Parse the email tokens and make a hash map of hash maps for the class passed as a parameter.
     * Example state of the map after this function: {Spam => ["orderviagra", 1000], ["viagratoday", 987]}
     * This function is for evaluation and should therefore be called on the evaluation set of emails.
     */
    public void buildBigramsHashMap(String docClass, List<String> emailTokens) {

        // Verify that the first parameter is valid
        if (!(docClass.equals("ham")) && !(docClass.equals("spam"))) {
            System.out.print("Error: Invalid class type. \n Options: 'spam, 'ham' ");
            return;
        }

        // Initialize the outer map for the document class.
        // If a hash map has not yet been initialized for this class, create one.
        // Otherwise, just grab the one that already exists.
        if (bayesMap.get(docClass) == null) {
            HashMap<String, Integer> classMap = new HashMap();
            bayesMap.put(docClass, classMap);
        }

        HashMap<String, Integer> curMap = bayesMap.get(docClass);

        for (int i = 0; i < emailTokens.size() - 1; i++) {
            String bigram = emailTokens.get(i) + emailTokens.get(i + 1);
            if (!curMap.containsKey(bigram)) {
                curMap.put(bigram, 0);
            }

            int curCount = curMap.get(bigram);
            curMap.put(bigram, curCount + 1);

        }

    }

    /*
     * Parse the tokens of the email passed as a parameter and sum the counts of each word.
     * Return the document class with the larger count.
     * This function is for evaluation and should therefore be called on the evaluation set of emails.
     */
    public String classifyWithBigrams(List<String> tokens) {

        HashMap<String, Integer> spamDist = bayesMap.get("spam");
        HashMap<String, Integer> hamDist = bayesMap.get("ham");

        double hamScore = 0;
        double spamScore = 0;

        for (int i = 0; i < tokens.size() - 1; i++) {
            String bigram = tokens.get(i) + tokens.get(i + 1);
            spamScore += Math.log(spamDist.getOrDefault(bigram, 1));
            hamScore += Math.log(hamDist.getOrDefault(bigram, 1));
        }

        if (hamScore > spamScore) {
            return "ham";
        }
        else {
            return "spam";
        }
    }

    /*
     * Parse the email tokens and make a hash map of hash maps for the class passed as a parameter.
     * Example state of the map after this function: {Spam => ["orderviagra", 1000], ["viagratoday", 987]}
     * This function is for evaluation and should therefore be called on the evaluation set of emails.
     */
    public void buildTrigramsHashMap(String docClass, List<String> emailTokens) {

        // Verify that the first parameter is valid
        if (!(docClass.equals("ham")) && !(docClass.equals("spam"))) {
            System.out.print("Error: Invalid class type. \n Options: 'spam, 'ham' ");
            return;
        }

        // Initialize the outer map for the document class.
        // If a hash map has not yet been initialized for this class, create one.
        // Otherwise, just grab the one that already exists.
        if (bayesMap.get(docClass) == null) {
            HashMap<String, Integer> classMap = new HashMap();
            bayesMap.put(docClass, classMap);
        }

        HashMap<String, Integer> curMap = bayesMap.get(docClass);

        for (int i = 0; i < emailTokens.size() - 2; i++) {
            String trigram = emailTokens.get(i) + emailTokens.get(i + 1) + emailTokens.get(i + 2);
            if (!curMap.containsKey(trigram)) {
                curMap.put(trigram, 0);
            }

            int curCount = curMap.get(trigram);
            curMap.put(trigram, curCount + 1);

        }

    }

    /*
     * Parse the tokens of the email passed as a parameter and sum the counts of each word.
     * Return the document class with the larger count.
     * This function is for evaluation and should therefore be called on the evaluation set of emails.
     */
    public String classifyWithTrigrams(List<String> tokens) {

        HashMap<String, Integer> spamDist = bayesMap.get("spam");
        HashMap<String, Integer> hamDist = bayesMap.get("ham");

        double hamScore = 0;
        double spamScore = 0;

        for (int i = 0; i < tokens.size() - 2; i++) {
            String trigram = tokens.get(i) + tokens.get(i + 1) + tokens.get(i + 2);
            spamScore += Math.log(spamDist.getOrDefault(trigram, 1));
            hamScore += Math.log(hamDist.getOrDefault(trigram, 1));
        }

        if (hamScore > spamScore) {
            return "ham";
        }
        else {
            return "spam";
        }
    }

}
