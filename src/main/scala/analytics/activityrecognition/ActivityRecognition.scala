package analytics.activityrecognition

import analytics.init.InitSparkCluster
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{LogisticRegressionModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.sql.DataFrame


object ActivityRecognition extends InitSparkCluster {

  val seed = conf.getString("hyperparameter.seed").toInt
  val metric =  conf.getString("hyperparameter.metric")
  val depth = conf.getString("hyperparameter.depth").toInt
  val trees = conf.getString("hyperparameter.trees").toInt
  val bin = conf.getString("hyperparameter.bin").toInt
  val impurity = conf.getString("hyperparameter.classifier.impurity2")
  val showRows = conf.getString("dataset.showRows")

  def main(args: Array[String]): Unit = {

    val usage =
      """
         Usage: ActivityRecognition [--sample-size double] [--to-parquet boolean] [--hyper-param boolean] [--file-path string]
           """
    if (args.length == 0) println(usage)
    val arglist = args.toList
    type OptionMap = Map[Symbol, Any]

    def nextOption(map: OptionMap, list: List[String]): OptionMap = {
      def isSwitch(s: String) = (s(0) == '-')

      list match {
        case Nil => map
        case "--sample-size" :: value :: tail =>
          nextOption(map ++ Map('samplesize -> value.toDouble), tail)
        case "--to-parquet" :: value :: tail =>
          nextOption(map ++ Map('toParquet -> value.toBoolean), tail)
        case "--file-path" :: value :: tail =>
          nextOption(map ++ Map('filepath -> value), tail)
        case "--hyper-param" :: value :: tail =>
          nextOption(map ++ Map('hyper -> value.toBoolean), tail)
        case option :: tail => map
      }
    }

    val options = nextOption(Map(), arglist)
    println(options)

    sqlContext.setConf("spark.sql.shuffle.partitions", conf.getString("spark.sql.shuffle.partitions"))

    val phoneAccCsv = conf.getString("phone.acc.csv")
    val phoneGyroCsv = conf.getString("phone.gyro.csv")

    val phoneAccParquet = conf.getString("phone.acc.parquet")
    val phoneGyroParquet = conf.getString("phone.gyro.parquet")

    val sampleSize = if (options.isDefinedAt('samplesize)) options.get('samplesize).get.asInstanceOf[Double] else conf.getString("dataset.fraction").toDouble
    val isConvertToParquet = if (options.isDefinedAt('toParquet)) options.get('toParquet).get.asInstanceOf[Boolean] else true
    val isTrainWithHyper = if (options.isDefinedAt('hyper)) options.get('hyper).get.asInstanceOf[Boolean] else false

    if(isConvertToParquet) convertToParquet(Array(phoneAccCsv, phoneGyroCsv), Array(phoneAccParquet, phoneGyroParquet))

    val ds = etlWithParquet(Array(phoneAccParquet, phoneGyroParquet))
    val sampleDs = ds.sample(false, sampleSize)

    if(isTrainWithHyper) trainRandomForestModelWithHyperparameters(sampleDs) else trainRandomForestModel(sampleDs)

    close

  }

  def convertToParquet(sourcePaths: Array[String], destinationPaths: Array[String]): Unit = {

    val phoneAccRaw = spark.read.option("header", "true").option("inferSchema", "true").csv(sourcePaths(0)).coalesce(32)
    val phoneGyroRaw = spark.read.option("header", "true").option("inferSchema", "true").csv(sourcePaths(1)).coalesce(32)

    phoneAccRaw.write.parquet(destinationPaths(0))
    phoneGyroRaw.write.parquet(destinationPaths(1))

    println("I converted the files to Parquet !!!")

  }


  def etlWithParquet(sourcePaths: Array[String]): DataFrame = {


    val phoneAcc = spark.read.option("header", "true").option("inferSchema", "true").parquet(sourcePaths(0))
    val phoneGyro = spark.read.option("header", "true").option("inferSchema", "true").parquet(sourcePaths(1))

    val phoneAccn = phoneAcc.drop("Arrival_Time")
      .drop("Creation_Time")
      .where("x is not null")
      .where("y is not null")
      .where("z is not null")
      .where("gt <> 'null'")
      .selectExpr("Index", "User", "Model", "Device", "x as x_a", "y as y_a", "z as z_a", "gt as gt_a")

    val phoneGyron = phoneGyro.drop("Arrival_Time")
      .drop("Creation_Time")
      .where("x is not null")
      .where("y is not null")
      .where("z is not null")
      .where("gt <> 'null'")
      .selectExpr("Index", "User", "Model", "Device", "x as x_g", "y as y_g", "z as z_g", "gt as gt_g")

    phoneAccn
      .join(phoneGyron, Seq("Index", "User", "Model", "Device"), "inner")
      .selectExpr("*", "gt_a = gt_g as gt_combi")
      .createOrReplaceTempView("vw_phone_acc_gyro")

    val phoneAccGyroEx = spark.sql(
      """     SELECT *, CASE
               WHEN gt_a is NULL AND gt_g is NOT NULL
                   THEN gt_g
               ELSE
                    CASE
                        WHEN gt_g is NULL AND gt_a is NOT NULL
                            THEN gt_a
                        ELSE
                           CASE
                               WHEN gt_g is NOT NULL AND gt_a is NOT NULL
                                 THEN gt_a
                            END
                    END
            END as gt_grand from vw_phone_acc_gyro         """)

    val phoneAccGyroCleaned = phoneAccGyroEx
      .where("gt_grand <> 'null' ")
      .where("gt_combi = true")

    phoneAccGyroCleaned.createOrReplaceTempView("vw_phone_acc_gyro_cleaned")

    println("Number of records for the ML stage :"+phoneAccGyroCleaned.count())

    phoneAccGyroCleaned.show(showRows.toInt)

    phoneAccGyroCleaned

  }


  def trainRandomForestModel(data: DataFrame): Unit = {

    println("Training Random Forest model")

    logger.info("Splitting training and test data ")
    val (trainingData,testData)=  splitAndCacheData(data)

    println("Number of records used for the ML trainingData stage :"+trainingData.count())

    logger.info("Creating pipeline ")
    val pipeline=createTrainingPipeline(data,false)

    logger.info("Creating model from pipeline - this is the training state :")
    val model = pipeline.fit(trainingData)

    logger.info("Making predictions :")
    val predictions = model.transform(testData)

    logger.info("Printing a sample of the predictions :")
    predictions.select("predicted_label", "gt_grand", "indexed_features").show(showRows.toInt)

    val evaluator = createEvaluator

    logger.info("Checking the accuracy")
    val accuracy = evaluator.evaluate(predictions)

    println("Test Error before hyper-parameter tuning = " + (1.0 - accuracy))

    val rm = new RegressionMetrics(
      predictions.select("prediction", "indexed_gt_grand").rdd.map(x =>
        (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double]))
    )

    println("MSE: " + rm.meanSquaredError)
    println("MAE: " + rm.meanAbsoluteError)
    println("RMSE Squared: " + rm.rootMeanSquaredError)
    println("R Squared: " + rm.r2)
    println("Explained Variance: " + rm.explainedVariance + "\n")
  }


  private def trainRandomForestModelWithHyperparameters(data: DataFrame) ={

    println("Training Random Forest model with hyper parameters and cross validation")

    val maxBinsRangeArray = Array(conf.getInt("hyperparameter.minBin"), conf.getInt("hyperparameter.maxBin"))
    val maxDepthRangeArray = Array(conf.getInt("hyperparameter.minDepth"), conf.getInt("hyperparameter.maxDepth"))
    val maxTreesRangeArray = Array(conf.getInt("hyperparameter.minNumOfTrees"), conf.getInt("hyperparameter.maxNumOfTrees"))
    val impurityArray = Array(conf.getString("hyperparameter.classifier.impurity1"), conf.getString("hyperparameter.classifier.impurity2"))

    logger.info("Splitting training and test data ")
    val (trainingData,testData)=  splitAndCacheData(data)

    println("Number of records used for the ML trainingData stage :"+trainingData.count())

    logger.info("Creating pipeline ")

    val pipeline = createTrainingPipeline(trainingData,true)
    val classifier =pipeline.getStages(2).asInstanceOf[RandomForestClassifier]


    val paramGrid = new ParamGridBuilder()
      .addGrid(classifier.maxBins, maxBinsRangeArray)
      .addGrid(classifier.maxDepth, maxDepthRangeArray)
      .addGrid(classifier.numTrees, maxTreesRangeArray)
      .addGrid(classifier.impurity, impurityArray)
      .build()

    val evaluator =createEvaluator

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(conf.getInt("hyperparameter.validation.numberOfFolds"))

    logger.info("Training with cross validation ")
    val pipelineFittedModel = cv.fit(trainingData)

    logger.info("Making predictions")
    val predictions= pipelineFittedModel.transform(testData)

    logger.info("Checking the accuracy")
    val accuracy = evaluator.evaluate(predictions)

    println("Test Error after hyper-parameter tuning = " + (1.0 - accuracy))

    println("Best model: avgMetrics ")
    pipelineFittedModel.avgMetrics.foreach(println)

    val bestPipelineModel = pipelineFittedModel.bestModel.asInstanceOf[PipelineModel]
    val stages = bestPipelineModel.stages

    println("Best model: parameters  ")
    stages(2).extractParamMap().toSeq.toList.foreach(println)

    val rm = new RegressionMetrics(
      predictions.select("prediction", "indexed_gt_grand").rdd.map(x =>
        (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double]))
    )

    println("MSE: " + rm.meanSquaredError)
    println("MAE: " + rm.meanAbsoluteError)
    println("RMSE Squared: " + rm.rootMeanSquaredError)
    println("R Squared: " + rm.r2)
    println("Explained Variance: " + rm.explainedVariance + "\n")

  }

  private  def createClassifier()={
    val classifier= new RandomForestClassifier()
      .setImpurity(impurity)
      .setMaxDepth(depth)
      .setNumTrees(trees)
      .setFeatureSubsetStrategy("auto")
      .setSeed(seed)
      .setLabelCol("indexed_gt_grand")
      .setFeaturesCol("indexed_features").setMaxBins(bin)
    classifier
  }

  private  def createClassifierForHyperParameters()={
    val classifier= new RandomForestClassifier()
      .setFeatureSubsetStrategy("auto")
      .setSeed(seed)
      .setLabelCol("indexed_gt_grand")
      .setFeaturesCol("indexed_features")
    classifier
  }


  private def createEvaluator()={
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexed_gt_grand")
      .setPredictionCol("prediction")
      .setMetricName(metric)
    evaluator

  }

  private def splitAndCacheData(data:DataFrame):(DataFrame,DataFrame) ={

    val trainingTestSplitArray= Array(conf.getString("dataset.split.training").toDouble, conf.getString("dataset.split.test").toDouble)
    val Array(trainingData, testData) = data.na.drop.randomSplit(trainingTestSplitArray, seed)

    trainingData.cache()
    testData.cache()

    (trainingData,testData)
  }

  private def createTrainingPipeline(data: DataFrame,isHyper:Boolean):Pipeline ={

    val featureCols = data.select("x_a", "y_a", "z_a", "x_g", "y_g", "z_g").columns

    logger.info("Creating labelIndexer :")

    val labelIndexer = new StringIndexer().setInputCol("gt_grand").setOutputCol("indexed_gt_grand").fit(data)

    logger.info("Creating Vector Assembler:")

    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("indexed_features")

    logger.info("Creating classifier :")

    val classifier =  if(isHyper) createClassifierForHyperParameters else createClassifier

    logger.info("Creating labelConverter :")

    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predicted_label")
      .setLabels(labelIndexer.labels)

    logger.info("Assembling the pipeline :")

    val pipeline = new Pipeline().setStages(Array(labelIndexer, assembler, classifier, labelConverter))
    pipeline
  }





}