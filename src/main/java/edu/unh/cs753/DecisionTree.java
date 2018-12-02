/*
 * DecisionTree.java
 * 
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
	

    private transient BayesCounter bayesCounter;
    private List<Email> trainEmails;
    private List<Email> testEmails;
    private Instances trainData;
    private Instances testData;
    private boolean addNBScoring;
    private LogitBoost lBoost;
    
    
    public DecisionTree(String trainFileLoc, String testFileLoc, boolean addNB) throws Exception {
	    
        super();
        initializeData(trainFileLoc, testFileLoc, addNB);
        
    }
    
    
    
    
    private void initializeData(String trainFileLoc, String testFileLoc, boolean addNB) throws IOException {
       
        trainEmails = new ArrayList<Email>();
        testEmails = new ArrayList<Email>();
        addNBScoring = addNB;
        lBoost = new LogitBoost();
        
        try {
         
            makeEmails(trainFileLoc, true);
            makeEmails(testFileLoc, false);
         
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
    
    
    private void setUpBayesCounter() throws IOException {
        bayesCounter = new BayesCounter();
        for(Email trainEmail : trainEmails) {
            if(trainEmail.label.equals("spam")) {
                bayesCounter.buildHashMap("spam", trainEmail.terms);
            }
            else {
                bayesCounter.buildHashMap("ham", trainEmail.terms);
            }
        }
    }
    
    
    
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
    
    
    
    private String convertCSVToArff(String csvFile, boolean train) throws Exception {
        
        String outFile = csvFile.substring(0, csvFile.indexOf('.'));
        
        String ret = outFile + ".arff";
        
        PrintWriter writer = new PrintWriter(ret, "UTF-8");
        writer.println("%% " + outFile + " arff file\n");
        writer.println("@relation spam_or_ham");
        writer.println("@attribute doc_length {<10, 10-500, >500}");
        writer.println("@attribute contains_spam_term {yes, no}");
        
        if(addNBScoring) {
            writer.println("@attribute spamVal NUMERIC");
            writer.println("@attribute hamVal NUMERIC");
        }
        
        writer.println("@attribute label {spam, ham}\n");
        writer.println("@data");
        
        
        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
            String line;
            
            while ((line = br.readLine()) != null) {
                
                Scanner scan = new Scanner(line);
                int cnt = 0;
                String label = "";
                String id = "";
                boolean isSpam = false;
        
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
                
                String out;
                
                if(addNBScoring) {
                    List<Double> vals = bayesCounter.getScores(tokens);
                    double hamVal = vals.get(0);
                    double spamVal = vals.get(1);
                    out = docLenNominal + ", " + 
                             containsSpamTermStr + ", " + spamVal + 
                             ", " + hamVal + ", " + label;
                }
                else {
                    out = docLenNominal + ", " + 
                             containsSpamTermStr + ", " + label;
                }
                
                writer.println(out);
            }
        }  
         
        writer.close();
        return ret;
    }
    
    
    private void makeEmails(String csvFile, boolean train) throws IOException {
        
        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
            String line;
            
            while ((line = br.readLine()) != null) {
                
                Scanner scan = new Scanner(line);
                int cnt = 0;
                String label = "";
                String id = "";
                boolean isSpam = false;
        
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
                if(train) {
                    trainEmails.add(email);
                }
                else {
                    testEmails.add(email);
                }
            }
        }
    }
    
    
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
    
    
    private class Email {
        
        String id;
        String label;
        List<String> terms;
        int docLen;
        boolean containsSpamTerm;
        
        
        Email(String id, String label, List<String> terms, boolean containsSpamTerm) {
            this.id = id;
            this.label = label; 
            this.terms = terms;
            this.docLen = terms.size();
            this.containsSpamTerm = containsSpamTerm;
        }
        
    }
    
    
    
}   // END DecisionTree.java

    
