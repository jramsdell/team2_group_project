

/**
 * LuceneIndexer.java
 * 
 * UNH CS753
 * Progrmming Assignment 2
 * Group 2
 * 
 * This class is responsible for creating the index composed of documents
 * and terms. 
 * 
**/


package edu.unh.cs753.indexing;

import edu.unh.cs.treccar_v2.Data;
import edu.unh.cs753.utils.IndexUtils;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.RAMDirectory;
import java.util.ArrayList;
import java.util.Scanner;
import java.io.*;
import java.io.IOException;
import java.io.FileNotFoundException;
import org.apache.lucene.index.IndexOptions;
import org.jsoup.Jsoup;
import javax.mail.*;
import javax.mail.internet.*;
import com.google.common.base.CharMatcher;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import java.util.List;
import java.util.Properties;
import java.nio.charset.Charset;
import java.nio.charset.CharsetEncoder;


public class LuceneIndexer {
    
    private final IndexWriter writer;

	/** 
	 * Construct a Lucene Indexer.
	 * @param indexLoc: the path containing the index.
    */
    public LuceneIndexer(String indexLoc) {
        writer = IndexUtils.createIndexWriter(indexLoc);
    }
    
    
    //--These 2 functions are from stack overflow---------------------------
    private String getTextFromMessage(Message message) throws MessagingException, IOException {
        String result = "";
        if(message.isMimeType("text/plain")) {
            result = message.getContent().toString();
        } 
        else if (message.isMimeType("multipart/*")) {
            MimeMultipart mmPart = (MimeMultipart) message.getContent();
            result = getTextFromMimeMultipart(mmPart);
        }
        
        return result;
    }

    private String getTextFromMimeMultipart(
        MimeMultipart mimeMultipart)  throws MessagingException, IOException{
        String result = "";
        int count = mimeMultipart.getCount();
        for (int i = 0; i < count; i++) {
            BodyPart bodyPart = mimeMultipart.getBodyPart(i);
            if(bodyPart.isMimeType("text/plain")) {
                result = result + "\n" + bodyPart.getContent();
                break; // without break same text appears twice in my tests
            } 
            else if (bodyPart.isMimeType("text/html")) {
                String html = (String) bodyPart.getContent();
                result = result + "\n" + org.jsoup.Jsoup.parse(html).text();
            } 
            else if (bodyPart.getContent() instanceof MimeMultipart){
                result = result + getTextFromMimeMultipart((MimeMultipart)bodyPart.getContent());
            }
        }
        return result;
    }
    //------------------------------------------------------------------
    
    
    // return a single string of tokens that have been run through Lucene Analyzer
    private String getAnalyzedTerms(String text) {
        String ret = "";
        Analyzer analyzer = new StandardAnalyzer();
        //Analyzer analyzer = new EnglishAnalyzer();
        try {
            TokenStream stream  = analyzer.tokenStream(null, new StringReader(text));
            stream.reset();
            while (stream.incrementToken()) {
                ret += stream.getAttribute(CharTermAttribute.class).toString() + " ";
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return ret.substring(0, ret.length()-1);
    }
    

    // this function iterates through all documents in corpus, uses MimeMessage class
    // to extract the terms, runs terms through the Analyzer and adds terms to index.
    public void doIndex(String docFilesPath) throws IOException, FileNotFoundException {
        
        // enable vector space model functionality
        FieldType ft = new FieldType();
        ft.setIndexOptions( IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS );
        ft.setStoreTermVectors( true );
        ft.setStoreTermVectorOffsets( true );
        ft.setStoreTermVectorPayloads( true );
        ft.setStoreTermVectorPositions( true );
        ft.setTokenized( true );
                
        File path = new File(docFilesPath);
        
        int counter = 0;
        int nErroneousFiles = 0;
        
        // FOR every file in corpus 
        File [] files = path.listFiles();
        for(int i = 0; i < files.length; i++) {
            if (files[i].isFile()) { 
                File filename = files[i];
  
                // create a MimeMessage class from the data in email file
                InputStream mailFileInputStream = new FileInputStream(filename);
                Properties props = new Properties();
                Session session = Session.getDefaultInstance(props, null);
                MimeMessage message = null;
                try {
                    message = new MimeMessage(session, mailFileInputStream);
                } catch(Exception e) {
                    System.out.println("CATCH 1: " + e);
                    System.exit(9);
                }
        
                String subject = "";
                
                try { // subject can be empty 
                    subject = getAnalyzedTerms(message.getSubject());
                } catch(Exception e) {}
                
                String text = ""; 
            
                try {
                    // extract the terms from the MimeMessage and run
                    // through Lucene's Analyzer
                    text = getAnalyzedTerms(getTextFromMessage(message).trim());

                    // Not sure if this is right, but for now I just add the
                    // subject to the body text (so it can be included in calculations)
                    String bodyText = subject + " " + text;
                    //System.out.println("bodyText: \"" + bodyText + "\"");
                    
                    Document doc = new Document();
                    String docID = filename.toString();
                    
                    // For now I am using the file name as document id,
                    // which also works for accessing the file in the ground truth file (index)
                    int index = docID.lastIndexOf('/');
                    docID = docID.substring(index+1, docID.length());
                    
                    doc.add(new StringField("id", docID, Field.Store.YES));
                    doc.add(new TextField("text", bodyText, Field.Store.YES));
                    
                    // add document to IndexWriter
                    writer.addDocument(doc);
                    // TODO: is this (counter, writer commit) still relevant/necessary?
                    counter++;
                    if(counter % 20 == 0) 
                        writer.commit();
                    
                }
                // NOTE: There were 134 out of 75k+ that threw exceptions, for
                // not starting with a valid ascii char (not sure if there's any 
                // way round this or if it's worth it to even try...)
                catch(Exception e) {
                    nErroneousFiles++;
                }
            }
        }
        
        writer.close();
        int nDocs = files.length;
        //System.out.println("Results: " + nErroneousFiles + ", " + nDocs);
        
    }

    

}
