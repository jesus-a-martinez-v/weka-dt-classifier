package net.datasmarts

fun main(args: Array<String>) {
    println("Loading data.")
    val data = Classifier.loadData()

    println("Splitting data.")
    val (train, test) = Classifier.trainTestSplit(data)

    println("Training unprunned decision tree classifier.")
    val model = Classifier.trainModel(train)

    println("Evaluating model.")
    Classifier.evaluateModel(model, train, test)
}

