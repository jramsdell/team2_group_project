import edu.unh.cs753.utils.parsing.EmailParsing

fun main(args: Array<String>) {

//    val emailParser = EmailParsing.readMail("/home/hcgs/data_science/data/spam/trec07p/data/inmail.1286")
    val emailContainer = EmailParsing.createEmailContainer("/home/hcgs/data_science/data/spam/trec07p/data/inmail.1286")
    println(emailContainer.subject)
    println(emailContainer.content)
//    emailParser.htmlEmailBody.`is`
//        .run {
//            EmailParsing.convertHtmlToPlainText(this.bufferedReader().readText()) }
//        .run { println(this) }
//        .bufferedReader()
//        .readLines()
//        .forEach { line -> println(line) }
}