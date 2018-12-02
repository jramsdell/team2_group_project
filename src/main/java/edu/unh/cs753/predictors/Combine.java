package edu.unh.cs753.predictors;

import edu.unh.cs753.utils.SearchUtils;
import org.apache.lucene.document.Document;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.similarities.BasicStats;
import org.apache.lucene.search.similarities.SimilarityBase;

import java.io.IOException;
import java.util.List;
import java.util.*;

public class Combine extends LabelPredictor{

    public Combine(IndexSearcher s) {
        super(s);

    }

    public SimilarityBase jm() {
        return new SimilarityBase() {
            @Override
            protected float score(BasicStats basicStats, float freq, float docLen) {

                if (docLen == 0) {
                    return 0 * basicStats.getBoost();
                } else {
                    float  corpus= (freq)/(docLen);
                    float jelenik= (float) ((corpus*0.1) + (freq*0.9));
                    return (float) Math.log(jelenik);
                }
            }

            @Override
            public String toString() {
                return null;
            }
        };
    }

    public SimilarityBase dirichlet() {
        return new SimilarityBase() {
            @Override
            protected float score(BasicStats basicStats, float freq, float docLen) {

                if (docLen == 0) {
                    return 0 * basicStats.getBoost();
                } else {
                    float mu = 1000;
                    float dr = 0;
                    dr = (freq + mu * basicStats.getNumberOfDocuments()) / (docLen + mu);
                    return dr;
                }
            }

            @Override
            public String toString() {
                return null;
            }
        };
    }

    @Override
    public String predict(List<String> tokens) {

        // The predict methods takens tokens from an email, and you must decide if the email is ham or spam using them.
        // Below is a stupid example where you just randomly assign a label.
        // You will want to replace this with your actual classification method
        List<String> tokenSubset = tokens.subList(0, Math.min(10, tokens.size()));
        Query q = SearchUtils.createStandardBooleanQuery(String.join(" ", tokenSubset), "text");
        float ham = 0;
        float spam = 0;
        float totalscore= 0;
        int counter = 0;

        try {
            TopDocs topDocs = searcher.search(q, 11);

            for ( ScoreDoc sd : topDocs.scoreDocs) {
                Document doc = searcher.doc(sd.doc);
                String label = doc.get("label");
                double invRank = (11 - counter) / 11.0;


                if (label.equals("ham")) {
                    ham += invRank;
                } else {
                    spam += invRank;
                }
                counter++;
            }


            searcher.setSimilarity(jm());
            topDocs = searcher.search(q, 11);
            counter = 0;

            for ( ScoreDoc sd : topDocs.scoreDocs) {
                Document doc = searcher.doc(sd.doc);
                String label = doc.get("label");
                double invRank = (11 - counter) / 11.0;


                if (label.equals("ham")) {
                    ham += invRank;
                } else {
                    spam += invRank;
                }
                counter++;
            }

            searcher.setSimilarity(dirichlet());
            topDocs = searcher.search(q, 11);
            counter = 0;

            for ( ScoreDoc sd : topDocs.scoreDocs) {
                Document doc = searcher.doc(sd.doc);
                String label = doc.get("label");
                double invRank = (11 - counter) / 11.0;


                if (label.equals("ham")) {
                    ham += invRank;
                } else {
                    spam += invRank;
                }
                counter++;
            }




        } catch (IOException e) {
            e.printStackTrace();
        }
        if (ham > spam)
        {
            return "ham";
        }
        else
        {
            return "spam";
        }
    }



    public static void main(String[] args) throws IOException {

        // This is where you test out your predictor by running evaluate
        // Make sure you created "parsed_emails.tsv" and the lucene index! See: DebugHelper
        IndexSearcher searcher = SearchUtils.createIndexSearcher("index");
        Combine combinescore= new Combine(searcher);
        combinescore.evaluate();
    }

}

