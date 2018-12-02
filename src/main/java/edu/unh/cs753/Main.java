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
			LabelPredictor predictor = null;

			if (method.equals("bayes")) {
				predictor = new NaiveBayesPredictor(searcher);
			} else if (method.equals("bayes_bigrams")) {
				 predictor = new NaiveBayesBigramPredictor(searcher);
			} else if (method.equals("bayes_trigram")) {
				 predictor = new NaiveBayesTrigramPredictor(searcher);
			} else if (method.equals("bayes_quadgram")) {
				 predictor = new NaiveBayesQuadgramPredictor(searcher);
			} else if (method.equals("kernel_embedding_4gram")) {
				 predictor = new SimpleKernelPredictor(searcher, ComponentRepresentation.FOURGRAM);
			} else if (method.equals("multiple_kernel_embedding_4gram")) {
				 predictor = new CombinedBasisPredictor(searcher, ComponentRepresentation.FOURGRAM);
			} else if (method.equals("kernel_embedding_unigram")) {
				 predictor = new SimpleKernelPredictor(searcher, ComponentRepresentation.UNIGRAM);
			} else if (method.equals("multiple_kernel_embedding_unigram")) {
				 predictor = new CombinedBasisPredictor(searcher, ComponentRepresentation.UNIGRAM);
			} else if (method.equals("kernel_embedding_bigram")) {
				 predictor = new SimpleKernelPredictor(searcher, ComponentRepresentation.BIGRAM);
			} else if (method.equals("multiple_kernel_embedding_bigram")) {
				 predictor = new CombinedBasisPredictor(searcher, ComponentRepresentation.BIGRAM);
			} else if (method.equals("lucene_bm25")) {
				 predictor = new BM25Predictor(searcher);
			} else if (method.equals("lucene_bm25_1")) {
				 predictor = new BM25Predictor1(searcher);
			} else if (method.equals("lucene_bm25_101")) {
				 predictor = new BM25Predictor101(searcher);
			} else if (method.equals("lucene_bm25_score")) {
				 predictor = new BM25Predictorscore(searcher);
			} else if (method.equals("lucene_combined")) {
				predictor = new Combine(searcher);
			} else if (method.equals("lucene_dirichlet")) {
				predictor = new Dirichlet(searcher);
			} else if (method.equals("lucene_inverse_rank")) {
				predictor = new Inverserank(searcher);
			} else if (method.equals("lucene_jm")) {
				predictor = new JM(searcher);
			} else if (method.equals("lucene_laplace")) {
				predictor = new Laplace(searcher);
			} else if (method.equals("bayes_knn")) {
				predictor = new KnnPredictor(searcher);
			} else if (method.equals("bayes_cosine")) {
				predictor = new CosinePredictor(searcher);
			} else if (method.equals("bayes_tfidf")) {
				predictor = new TfidfPredictor(searcher);
			}

			if (predictor != null) {
				predictor.evaluate();
			} else {
				System.out.println("Unknown classification method!: " + method);
			}
		}

	}

}
