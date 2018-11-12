package edu.unh.cs753;

import evaluation.KotlinEmailParser;
import evaluation.KotlinNaiveBayesTrainer;
import evaluation.KotlinTrainingEmailIndexer;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;



public class Main {

	// Main class for project
	public static void main(String[] args) throws IOException {
		System.setProperty("file.encoding", "UTF-8");

		String option = args[0];
		String path = args[1];

		if (option.equals("parse")) {
			KotlinEmailParser.INSTANCE.parseEmails(path);
		} else if (option.equals("index")) {
			KotlinTrainingEmailIndexer.INSTANCE.createIndex(path, "index");
		} else if (option.equals("classify")) {
			KotlinNaiveBayesTrainer trainer = new KotlinNaiveBayesTrainer();
			trainer.doTrain(path);
		} else if (option.equals("tree")) {
			DecisionTree tree = new DecisionTree(path);
			// ...
		}

	}

}
