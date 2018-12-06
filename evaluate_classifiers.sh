echo Naive Bayes Baseline
java -jar target/team2_group_project-1.0-SNAPSHOT-jar-with-dependencies.jar classify bayes
echo

echo Bigram Bayes
java -jar target/team2_group_project-1.0-SNAPSHOT-jar-with-dependencies.jar classify bayes_bigrams
echo

echo Trigram Bayes
java -jar target/team2_group_project-1.0-SNAPSHOT-jar-with-dependencies.jar classify bayes_trigram
echo

echo Quadgram Bayes
java -jar target/team2_group_project-1.0-SNAPSHOT-jar-with-dependencies.jar classify bayes_quadgram
echo



echo TFIDF Bayes
java -jar target/team2_group_project-1.0-SNAPSHOT-jar-with-dependencies.jar classify bayes_tfidf
echo

echo Kernel Embedding 5Gram
java -jar target/team2_group_project-1.0-SNAPSHOT-jar-with-dependencies.jar classify kernel_embedding_4gram
echo

echo Kernel Embedding Unigram
java -jar target/team2_group_project-1.0-SNAPSHOT-jar-with-dependencies.jar classify kernel_embedding_unigram
echo

echo Kernel Embedding Bigram
java -jar target/team2_group_project-1.0-SNAPSHOT-jar-with-dependencies.jar classify kernel_embedding_bigram
echo

echo Lucene BM25
java -jar target/team2_group_project-1.0-SNAPSHOT-jar-with-dependencies.jar classify lucene_bm25
echo

echo Lucene BM25 Top 1 Doc
java -jar target/team2_group_project-1.0-SNAPSHOT-jar-with-dependencies.jar classify lucene_bm25_1
echo

echo Lucene BM25 Top 101 Docs
java -jar target/team2_group_project-1.0-SNAPSHOT-jar-with-dependencies.jar classify lucene_bm25_101
echo

echo Lucene BM25 Using Doc Scores
java -jar target/team2_group_project-1.0-SNAPSHOT-jar-with-dependencies.jar classify lucene_bm25_score
echo

echo Lucene BM25 Sum of Inverse rank: BM25 + JM + Laplace
java -jar target/team2_group_project-1.0-SNAPSHOT-jar-with-dependencies.jar classify lucene_combined
echo

echo Lucene Laplace
java -jar target/team2_group_project-1.0-SNAPSHOT-jar-with-dependencies.jar classify lucene_laplace
echo

echo Lucene Dirichlet
java -jar target/team2_group_project-1.0-SNAPSHOT-jar-with-dependencies.jar classify lucene_dirichlet
echo

echo Lucene JM
java -jar target/team2_group_project-1.0-SNAPSHOT-jar-with-dependencies.jar classify lucene_jm
echo

echo Lucene Sum of BM25 Inverse Rank
java -jar target/team2_group_project-1.0-SNAPSHOT-jar-with-dependencies.jar classify lucene_inverse_rank
echo
