package edu.unh.cs753.utils.parsing;

//import java.util.List;

public class EmailContainer {
        public String subject;
        public String content;
        public List<String> tokens;

        public EmailContainer(String sub, String cont, String label, List<String> tokens) {
                subject = sub;
                content = cont;
                tokens = tokens;

        }
}
