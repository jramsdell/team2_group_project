package edu.unh.cs753.utils.parsing;

import org.apache.commons.io.IOUtils;
import org.apache.james.mime4j.MimeException;
import org.apache.james.mime4j.codec.DecodeMonitor;
import org.apache.james.mime4j.message.DefaultBodyDescriptorBuilder;
import org.apache.james.mime4j.parser.ContentHandler;
import org.apache.james.mime4j.parser.MimeStreamParser;
import org.apache.james.mime4j.stream.BodyDescriptorBuilder;
import org.apache.james.mime4j.stream.MimeConfig;
import org.jsoup.Jsoup;
import tech.blueglacier.email.Email;
import tech.blueglacier.parser.CustomContentHandler;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;

import java.io.StringReader;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import java.util.List;
import java.util.ArrayList;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;


public class EmailParsing {


    public static EmailContainer createEmailContainer(String emailFileLoc) {
            try {
                Email email = readMail(emailFileLoc);
                String subject = email.getEmailSubject();
                String html = IOUtils.toString(email.getHTMLEmailBody().getIs());
                String content = convertHtmlToPlainText(html);
                int index = emailFileLoc.lastIndexOf('/');
                String docID = emailFileLoc.substring(index+1, emailFileLoc.length());
                List<String> list = getAnalyzedTerms(content);
                
                return new EmailContainer(docID, subject, content, "", list);
            } catch (IOException e) {
                e.printStackTrace();
            }

            return null;
    }

    public static Email readMail(String filePath) throws IOException {
        ContentHandler contentHandler = new CustomContentHandler();

        MimeConfig mime4jParserConfig = MimeConfig.DEFAULT;
        BodyDescriptorBuilder bodyDescriptorBuilder = new DefaultBodyDescriptorBuilder();
        MimeStreamParser mime4jParser = new MimeStreamParser(mime4jParserConfig, DecodeMonitor.SILENT,bodyDescriptorBuilder);
        mime4jParser.setContentDecoding(true);
        mime4jParser.setContentHandler(contentHandler);
        FileInputStream fs = new FileInputStream(filePath);
        BufferedInputStream bs = new BufferedInputStream(fs);

        try {
            mime4jParser.parse(bs);
        } catch (MimeException e) {
            e.printStackTrace();
        }

        Email email = ((CustomContentHandler) contentHandler).getEmail();
        return email;
    }

    public static String convertHtmlToPlainText(String htmlString) {
        return Jsoup.parse(htmlString).text();
    }

    private static List<String> getAnalyzedTerms(String text) {
        List<String> ret = new ArrayList<String>();
        Analyzer analyzer = new EnglishAnalyzer();
        //Analyzer analyzer = new StandardAnalyzer();
       
        try {
            TokenStream stream  = analyzer.tokenStream(null, new StringReader(text));
            stream.reset();
            while (stream.incrementToken())
                ret.add(stream.getAttribute(CharTermAttribute.class).toString());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return ret;
    }



}


