/*
 * DecisionTree.java
 * 
 */

package edu.unh.cs753;


import java.io.BufferedReader;
import java.io.FileReader;
import weka.core.Instances;
import weka.core.Attribute;
import weka.core.converters.ArffLoader;
import weka.classifiers.Evaluation;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;


public class DecisionTree {
	
    
    public DecisionTree(String path) {
        
        ArffLoader loader= new ArffLoader();
        // TODO: how to gather informtion for the arff file
        //loader.setSource(new File(arffFile));
        Instances data; // = loader.getDataSet();
        
        Classifier cls = new J48();
        
        /*
        BufferedReader trainReader = new BufferedReader(new FileReader(arffFile));
        Instances train = new Instances(trainReader); 
        train.setClass(data.attribute("?"));
        
        BufferedReader testReader = new BufferedReader(new FileReader(arffFile));
        Instances test = new Instances(testReader);  
        test.setClass(data.attribute("?")); 
        
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(cls, test);
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        */
        
    }
    
}

