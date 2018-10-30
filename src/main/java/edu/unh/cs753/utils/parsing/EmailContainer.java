package edu.unh.cs753.utils.parsing;

import java.util.List;

public class EmailContainer {
        public String filename;
        public String subject;
        public String content;
        public List<String> tokens;

        public EmailContainer(String filename, String sub, String cont, String label, List<String> tokens) {
                filename = filename;
                subject = sub;
                content = cont;
                tokens = tokens;
        }
}
