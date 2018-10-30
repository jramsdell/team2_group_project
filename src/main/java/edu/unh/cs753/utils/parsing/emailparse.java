package edu.unh.cs753.utils.parsing;


import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;


public class emailparse {

    public static Map<String,String> readfilehamfull() throws IOException {

        //     ArrayList<String> ham = new ArrayList<>();
        Map<String,String> ham= new HashMap<>();


        BufferedReader br1 = new BufferedReader(new FileReader("/Users/abnv/Desktop/trec07p/full/index"));

        while (true) {

            String line1 = br1.readLine();

            if (line1 == null) {

                break;
            }

            String splitwords1[] = line1.split(" ");


            if (splitwords1[0].toLowerCase().equals("ham"))
            {
                String inner[]= splitwords1[1].split("/");
                ham.put(inner[2],"ham");

            }


        }
        return ham;
    }


    public static Map<String, String> readfilespam() throws IOException {

        Map<String, String> spam = new HashMap<>();

        BufferedReader br1 = new BufferedReader(new FileReader("/Users/abnv/Desktop/trec07p/full/index"));

        while (true) {

            String line1 = br1.readLine();

            if (line1 == null) {

                break;
            }

            String splitwords1[] = line1.split(" ");

            if (splitwords1[0].toLowerCase().equals("ham"))

            {

                String inner1[]=splitwords1[1].split("/");

                spam.put(inner1[2],"spam");
            }


        }
        return spam;
    }
}