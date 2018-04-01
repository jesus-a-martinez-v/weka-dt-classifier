package net.datasmarts

import weka.classifiers.AbstractClassifier
import weka.classifiers.Evaluation
import weka.classifiers.trees.J48
import weka.core.Instances
import weka.core.converters.ConverterUtils
import java.util.*
import kotlin.math.round

object Classifier {
    fun loadData(): Instances {
        val filePath = this.javaClass.getResource("/iris.csv").toURI().path
        val source = ConverterUtils.DataSource(filePath)
        val data = source.dataSet

        val unknownClassIndex = data.classIndex() == -1

        if (unknownClassIndex) {
            println("Setting class index.")
            data.setClassIndex(data.numAttributes() - 1)
        }

        return data
    }

    fun trainTestSplit(data: Instances, trainProportion: Double = 0.8, seed: Long = 42): Pair<Instances, Instances> {
        require(0 < trainProportion && trainProportion < 1, { "Train proportion must be between 0 and 1." })
        data.randomize(Random(seed))

        val trainSize = round(data.numInstances() * trainProportion).toInt()
        val testSize = data.numInstances() - trainSize

        val train = Instances(data, 0, trainSize)
        val test = Instances(data, trainSize, testSize)

        return Pair(train, test)
    }

    fun trainModel(instances: Instances): AbstractClassifier {
        val tree = J48()
        tree.unpruned = true

        tree.buildClassifier(instances)

        return tree
    }

    fun evaluateModel(model: AbstractClassifier, train: Instances, test: Instances) {
        val evaluator = Evaluation(train)

        evaluator.evaluateModel(model, test)
        val summary = evaluator.toSummaryString("Results", false)

        println(summary)
    }
}