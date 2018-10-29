package edu.unh.cs753;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;


public class filter {


    public static ArrayList<String> readfilespamfull() throws IOException {

        ArrayList<String> spam = new ArrayList<>();

        BufferedReader br = new BufferedReader(new FileReader("/Users/abnv/Desktop/trec07p/full/index"));

        while (true) {

            String line = br.readLine();

            if (line == null) {

                break;
            }


            String splitwords[] = line.split(" ");
            if (splitwords[0].toLowerCase().equals("spam")) {
                spam.add(splitwords[1]);
            }



        }
        return spam;
//        for(String d:spam){
//            System.out.println(d);
//        }
    }


    public static ArrayList<String> readfilehamfull() throws IOException {

        ArrayList<String> ham = new ArrayList<>();

        BufferedReader br1 = new BufferedReader(new FileReader("/Users/abnv/Desktop/trec07p/full/index"));

        while (true) {

            String line1 = br1.readLine();

            if (line1 == null) {

                break;
            }

            String splitwords1[] = line1.split(" ");

            if (splitwords1[0].toLowerCase().equals("ham")) {


                ham.add(splitwords1[1]);
            }


        }
        return ham;
    }
}