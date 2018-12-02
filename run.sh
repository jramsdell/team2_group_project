#!/bin/bash



# maven compile command
mvn clean compile assembly:single
if [ $? -eq 1 ]; then
    echo "Maven compile failed"
    exit 1
fi


# run the maven jar file (to display results)
java -jar target/team2_group_project-1.0-SNAPSHOT-jar-with-dependencies.jar $1 $2 $3 $4
if [ $? -eq 1 ]; then
    echo "Error running jar file"
    exit 1
fi
