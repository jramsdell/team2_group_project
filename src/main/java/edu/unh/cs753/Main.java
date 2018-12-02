package edu.unh.cs753;

import components.ComponentRepresentation;
import edu.unh.cs753.predictors.*;
import edu.unh.cs753.utils.SearchUtils;
import evaluation.KotlinEmailParser;
import evaluation.KotlinNaiveBayesTrainer;
import evaluation.KotlinTrainingEmailIndexer;
import org.apache.lucene.search.IndexSearcher;
import predictors.CombinedBasisPredictor;
import predictors.SimpleKernelPredictor;

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
			String method = args[1];
			IndexSearcher searcher = SearchUtils.createIndexSearcher("index");

			if (method.equals("bayes")) {
				LabelPredictor predictor = new NaiveBayesPredictor(searcher);
				predictor.evaluate();
			} else if (method.equals("bayes_bigrams")) {
				LabelPredictor predictor = new NaiveBayesBigramPredictor(searcher);
				predictor.evaluate();
			} else if (method.equals("bayes_trigram")) {
				LabelPredictor predictor = new NaiveBayesTrigramPredictor(searcher);
				predictor.evaluate();
			} else if (method.equals("bayes_quadgram")) {
				LabelPredictor predictor = new NaiveBayesQuadgramPredictor(searcher);
				predictor.evaluate();
			} else if (method.equals("kernel_embedding_4gram")) {
				LabelPredictor predictor = new SimpleKernelPredictor(searcher, ComponentRepresentation.FOURGRAM);
				predictor.evaluate();
			} else if (method.equals("multiple_kernel_embedding_4gram")) {
				LabelPredictor predictor = new CombinedBasisPredictor(searcher, ComponentRepresentation.FOURGRAM);
				predictor.evaluate();
			} else if (method.equals("kernel_embedding_unigram")) {
				LabelPredictor predictor = new SimpleKernelPredictor(searcher, ComponentRepresentation.UNIGRAM);
				predictor.evaluate();
			} else if (method.equals("multiple_kernel_embedding_unigram")) {
				LabelPredictor predictor = new CombinedBasisPredictor(searcher, ComponentRepresentation.UNIGRAM);
				predictor.evaluate();
			} else if (method.equals("kernel_embedding_bigram")) {
				LabelPredictor predictor = new SimpleKernelPredictor(searcher, ComponentRepresentation.UNIGRAM);
				predictor.evaluate();
			} else if (method.equals("multiple_kernel_embedding_bigram")) {
				LabelPredictor predictor = new CombinedBasisPredictor(searcher, ComponentRepresentation.UNIGRAM);
				predictor.evaluate();
			}
		}

	}

}
