# team2_group_project

## Dataset

This project utilizes the TREC 2007 SPAM corpus. You may access the data through the site: https://plg.uwaterloo.ca/~gvcormac/treccorpus07/

Download the 255 MB corpus, unzip it, and make note of its path.

## Results

If you are interested in the evaluation results, refer to report.pdf. It contains a summary of our methods used to implement the baseline, and the evaluation score for this baseline.

## Installation

After cloning the repo, change into the project directory and run the compile script (./compile.sh)
This will create a jar file at: target/team2_group_project-1.0-SNAPSHOT-jar-with-dependencies.jar

## Parsing emails

The emails must first be parsed into a tsv format. You can do this by running the following command while in the project directory:

java -jar target/team2_group_project-1.0-SNAPSHOT-jar-with-dependencies.jar parse SPAMDIR

Where SPAMDIR is the path to the unzipped spam directory (the directory should be called trec07p).

This will create a parsed_emails.tsv file located within the project directory. You will need this for classification.

## Classification

You can classify the emails by running the following command while in the project directory:

java -jar target/team2_group_project-1.0-SNAPSHOT-jar-with-dependencies.jar classify parsed_emails.tsv

This assumes that you parsed the TREC emails (described in the previous section).
