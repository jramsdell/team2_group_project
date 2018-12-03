/*
 * DecisionTree.java
 * Tim Ward
 * CS853 UNH
 * 
 * This is an implementation of a decision tree using weka. It is currently using 
 * a random forest, which is a combination of tree predictors. This 
 * implementation also works with the J48 tree. The chosen features
 * are document length and whether or not it contains certain terms 
 * indicitive of spam. Adding our naive bayes ham and spam scores as
 * a feature is optional and greatly increases the correctly classified
 * documents. Unigram, bigram, trigram and quadgram naive bayes scoring
 * was used. 
 * 
 */

package edu.unh.cs753;


import java.io.BufferedReader;
import java.io.FileReader;
import java.io.File;
import java.io.PrintWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.Files;
import java.util.Scanner;
import java.util.Random;
import java.util.List;
import java.util.ArrayList;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.HashMap;
import java.awt.BorderLayout;
import weka.core.Instance;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.Attribute;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.J48;
import weka.classifiers.meta.LogitBoost;



public class DecisionTree extends RandomForest {
	

    private BayesCounter bayesCounter;
    private List<Email> trainEmails;
    private List<Email> testEmails;
    private Instances trainData;
    private Instances testData;
    private boolean addNBScoring;
    private LogitBoost lBoost;
    private HashMap<String, AtomicInteger> docFreqs;
    private int nDocs;
    
    
    public DecisionTree(String trainFileLoc, String testFileLoc, boolean addNB) throws Exception {
        super();
        initializeData(trainFileLoc, testFileLoc, addNB);
    }
    
    
    
    /*
     * Initialize the decision tree, which consists of setting up the
     * bayes counter, converting the csv (or tsv) files to arff format,
     * setting up the training and test instances and building the classifier.
     */
    private void initializeData(String trainFileLoc, String testFileLoc, boolean addNB) throws IOException {
       
        nDocs = 0;
        trainEmails = new ArrayList<Email>();
        testEmails = new ArrayList<Email>();
        addNBScoring = addNB;
        lBoost = new LogitBoost();
        docFreqs = new HashMap<String, AtomicInteger>();
        
        try {
         
            makeEmails(trainFileLoc, true);
            makeEmails(testFileLoc, false);
            
            setDocFreqs();
            
            nDocs = trainEmails.size() + testEmails.size();
         
            if(addNB) {
                setUpBayesCounter();
            }
         
            String trainArffFilename = convertCSVToArff(trainFileLoc, true);
            String testArffFilename = convertCSVToArff(testFileLoc, false);
                
            DataSource trainSource = new DataSource(trainArffFilename);
            trainData = trainSource.getDataSet();
            trainData.setClassIndex(trainData.numAttributes() - 1);

            DataSource testSource = new DataSource(testArffFilename);
            testData = testSource.getDataSet();
            testData.setClassIndex(testData.numAttributes() - 1);

            buildClassifier(trainData);
            lBoost.buildClassifier(trainData);
            
        } catch(Exception e) {
            System.err.println("initializeData FAILED: " + e);
            e.printStackTrace(System.err);
        } 
    } 
    
    
    /*
     * Evaluate the decision tree, which uses the train data to
     * predict the test set classification
     */
    public void evaluateModel() {
        try {
            Evaluation eval = new Evaluation(trainData); 
            //eval.evaluateModel(lBoost, testData);
            eval.evaluateModel(this, testData);
            /** Print the algorithm summary */
            System.out.println("====Results=====");
            System.out.println(eval.toSummaryString());
            System.out.println(this);
            System.out.println(eval.toMatrixString());
            System.out.println(eval.toClassDetailsString());
            
        } catch(Exception e) {
            System.err.println("Error evaluating model: " + e);
            e.printStackTrace(System.err);
        }
    }
    
    
    /*
     * Set up the bayes counter to incorporate naive bayes into the
     * features of the decision tree.
     */
    private void setUpBayesCounter() throws IOException {
        bayesCounter = new BayesCounter();
        for(Email trainEmail : trainEmails) {
            if(trainEmail.label.equals("spam")) {
                bayesCounter.buildHashMap("spam", trainEmail.terms);
                //bayesCounter.buildBigramsHashMap("spam", trainEmail.terms);
                //bayesCounter.buildTrigramsHashMap("spam", trainEmail.terms);
                //bayesCounter.buildQuadgramsHashMap("spam", trainEmail.terms);
            }
            else {
                bayesCounter.buildHashMap("ham", trainEmail.terms);
                //bayesCounter.buildBigramsHashMap("ham", trainEmail.terms);
                //bayesCounter.buildTrigramsHashMap("ham", trainEmail.terms);
                //bayesCounter.buildQuadgramsHashMap("ham", trainEmail.terms);
            }
        }  
    }
    
    
    /*
     * set the document frequency hash map necessary for tf-idf scoring.
     */ 
    private void setDocFreqs() {
        
        for(Email trainEmail : trainEmails) {
            HashMap<String, String> curDocMap = new HashMap<String, String>();
            for(String token : trainEmail.terms) {
                if(!curDocMap.containsKey(token)) {
                    curDocMap.put(token, token);
                    if(!docFreqs.containsKey(token)) {
                        docFreqs.put(token, new AtomicInteger(1));
                    }
                    else {
                        docFreqs.get(token).incrementAndGet();
                    }
                }
            }
        }
        
        for(Email testEmail : testEmails) {
            HashMap<String, String> curDocMap = new HashMap<String, String>();
            for(String token : testEmail.terms) {
                if(!curDocMap.containsKey(token)) {
                    curDocMap.put(token, token);
                    if(!docFreqs.containsKey(token)) {
                        docFreqs.put(token, new AtomicInteger(1));
                    }
                    else {
                        docFreqs.get(token).incrementAndGet();
                    }
                }
            }
        }
    }
    
    
    /*
     * Retrieve tf-idf score of terms in a document.
     */
    private float emailTfIdfScore(Email email) {
        float ret = 0;
        for(String token : email.terms) {
            int freq = email.termFreqs.get(token).intValue();
            int df = docFreqs.get(token).intValue();
            ret += (float)(Math.log(freq) * Math.log(nDocs/df));
        }
        return ret;
    }
    
    
    /*
     *  Convert a csv or tsv file to an arff file, which weka needs to
     *  build decision trees. An arff file is an attribute column based format
     */
    private String convertCSVToArff(String csvFile, boolean train) throws Exception {
        
        String outFile = csvFile.substring(0, csvFile.indexOf('.'));
        
        String ret = outFile + ".arff";
        
        PrintWriter writer = new PrintWriter(ret, "UTF-8");
        writer.println("%% " + outFile + " arff file\n");
        writer.println("@relation spam_or_ham");
        writer.println("@attribute doc_length {<10, 10-500, >500}");
        writer.println("@attribute contains_spam_term {yes, no}");
        writer.println("@attribute tfIdf NUMERIC");
        if(addNBScoring) {
            writer.println("@attribute spamVal NUMERIC");
            writer.println("@attribute hamVal NUMERIC");
        }
        // make class attribute (label) the last attribute (arff convention)
        writer.println("@attribute label {spam, ham}\n");
        writer.println("@data");
        
        List<Email> emails;
        if(train) {
            emails = trainEmails;
        }
        else {
            emails = testEmails;
        }
                
        for( Email email : emails) {  
            
            List<String> tokens = email.terms;
            
            int docLen = email.docLen;
            String docLenNominal;

            if(docLen < 10) {
                docLenNominal = "<10";
            }
            else if(docLen <= 500) {
                docLenNominal = "10-500";
            }
            else {
                docLenNominal = ">500"; 
            }      
                
            String containsSpamTermStr; 
            if(emailContainsSpamTerm(tokens)) { 
                containsSpamTermStr = "yes";
            }
            else {
                containsSpamTermStr = "no";
            }    

            float tfIdf = emailTfIdfScore(email);
            String label = email.label;  
            String out;
                
            if(addNBScoring) {
                List<Double> vals = bayesCounter.getScores(tokens);
                //List<Double> vals = bayesCounter.getBigramScores(tokens);
                //List<Double> vals = bayesCounter.getTrigramScores(tokens);
                //List<Double> vals = bayesCounter.getQuadramScores(tokens);
                double hamVal = vals.get(0);
                double spamVal = vals.get(1);      
                out = docLenNominal + ", " + containsSpamTermStr + 
                    ", " + tfIdf + ", " + spamVal + ", " + hamVal + ", " + label;
            }
            else {
                out = docLenNominal + ", " + 
                             containsSpamTermStr + ", " + tfIdf + ", " + label;
            }
            writer.println(out);
        }
          
        writer.close();
        return ret;
    }
    
    
    /*
     * Make an email container object that contains it's features and text
     */
    private void makeEmails(String csvFile, boolean train) throws IOException {
        
        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
            String line;
            while ((line = br.readLine()) != null) {
                Scanner scan = new Scanner(line);
                int cnt = 0;
                String label = "";
                String id = "";
                boolean isSpam = false;
                HashMap<String, AtomicInteger> freqs = new HashMap<String, AtomicInteger>();
                List<String> tokens = new ArrayList<String>();
                while(scan.hasNext()) {
                    String token = scan.next();
                    if(cnt == 0) {
                        id = token;
                    }
                    else if(cnt == 1) {
                        label = token;
                        if(label.equals("spam")) {
                            isSpam = true;
                        }
                        else {
                            isSpam = false;
                        }
                    }
                    else if(cnt > 1) {
                        tokens.add(token);
                        if(!freqs.containsKey(token)) {
                            freqs.put(token, new AtomicInteger(1));
                        }
                        else {
                            freqs.get(token).incrementAndGet();
                        }
                    }
                    cnt++;
                }
                
                int docLen = tokens.size();
                String docLenNominal;

                if(docLen < 10) {
                    docLenNominal = "<10";
                }
                else if(docLen <= 500) {
                    docLenNominal = "10-500";
                }
                else {
                    docLenNominal = ">500"; 
                }
                
                String containsSpamTermStr; 
                boolean containsSpamTerm;
                if(emailContainsSpamTerm(tokens)) { 
                    containsSpamTermStr = "yes";
                    containsSpamTerm = true;
                }
                else {
                    containsSpamTermStr = "no";
                    containsSpamTerm = false;
                }

                Email email = new Email(id, label, tokens, containsSpamTerm);
                email.setTermFreqs(freqs);
                if(train) {
                    trainEmails.add(email);
                }
                else {
                    testEmails.add(email);
                }
            }
        }
    }
    
    
    /*
     * Check to see if an email contains spam keywords
     */
    private boolean emailContainsSpamTerm(List<String> tokens) {
        
        final String spamTerms[] = {
            "viagra", "discount", "click" 
        };
        
        for(String token : tokens) {
            for(String spamTerm : spamTerms) {
                if(spamTerm.equals(token)) {
                    return true;
                }
            }
        }
        return false;
    }
    
    
    /*
     * For the document length feature, used to test bounds for spam
     * and ham document length estimation.
     */
    private int getSpamDocLenAvg() {
        
        int spamAvg = 0;
        int maxLen = 0;
        int zeroToTen = 0;
        int elevenTo500 = 0;
        int gt500 = 0;
        
        for(Email trainEmail : trainEmails) {
            if(trainEmail.label.equals("spam")) {
                int docLen = trainEmail.docLen;
                spamAvg += docLen;
                if(docLen > maxLen) {
                    maxLen = docLen;
                }
                if(docLen < 11) {
                    zeroToTen++;
                }
                else if(docLen > 400) {
                    gt500++;
                }
                else {
                    elevenTo500++;
                }
            }
        }
        spamAvg /= trainEmails.size();
        return spamAvg;
    }
    
    
    /*
     * Email private class is a container that contains all of the 
     * features in a document.
     */
    private class Email {
        
        String id;
        String label;
        List<String> terms;
        int docLen;
        boolean containsSpamTerm;
        HashMap<String, AtomicInteger> termFreqs;
        
        Email(String id, String label, List<String> terms, boolean containsSpamTerm) {
            this.id = id;
            this.label = label; 
            this.terms = terms;
            this.docLen = terms.size();
            this.containsSpamTerm = containsSpamTerm;
        }
        
        // add the term frequency map of the email
        void setTermFreqs(HashMap<String, AtomicInteger> map) {
            this.termFreqs = map;
        }
        
    }
}   // END DecisionTree.java

    
