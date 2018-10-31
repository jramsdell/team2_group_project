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

public class EmailParsing {


    public static EmailContainer createEmailContainer(String emailFileLoc) {
            try {
                Email email = readMail(emailFileLoc);
                String subject = email.getEmailSubject();
                String html = IOUtils.toString(email.getHTMLEmailBody().getIs());
                String content = convertHtmlToPlainText(html);
//                return new EmailContainer(subject, content, ""); // you should not push code with errors...
                return null;
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


    public static List<EmailContainer> parseCorpus(String corpusLoc) {
        ArrayList<EmailContainer> ret = new ArrayList<EmailContainer>();
        File path = new File(corpusLoc);
        File [] files = path.listFiles();
        for(int i = 0; i < files.length; i++) {
            if(files[i].isFile()) { 
                String filename = files[i].toString();
                //EmailContainer ec = createEmailContainer(filename);
                //ret.add(ec);
            }
        }
        return ret;
    }


}


