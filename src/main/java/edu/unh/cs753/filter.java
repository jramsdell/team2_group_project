

package edu.unh.cs753;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.regex.*;
public class filter {

void readFile() throws IOException {

    ArrayList<String> spam=new ArrayList<>();
    ArrayList<String> ham=new ArrayList<>();

    BufferedReader b= new BufferedReader(new FileReader("/Users/abnv/Desktop/trec07p/full/index"));

    while(true)
    {

        String line = b.readLine();

        if (line == null) {
            break;
        }

        String splitWords[] = line.split(" ");
        if(splitWords[0].toLowerCase().equals("spam"))
        {
            spam.add(splitWords[1]);
        }
        else
        {
            ham.add(splitWords[1]);
        }

    }

    System.out.println("The Spam is as follows");

    for(String d:spam){

        System.out.println(d);

    }

//    for(int i=0;i<spam.size();i++)
//    {
//        System.out.println(spam.get(i));
//    }


    System.out.println("The Ham is as follows");

    for(String d:ham){

        System.out.println(d);
    }


}


}
