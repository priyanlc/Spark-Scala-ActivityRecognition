package analytics.activityrecognition

import analytics.common.InitSparkCluster
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.{IndexToString, StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.mllib.evaluation.RegressionMetrics



object ActivityRecognition extends InitSparkCluster {

  //Set logger level to Warn
  Logger.getRootLogger.setLevel(Level.WARN)

  case class phoneAccCoordinates(index:Int,user:String,x:Double,y:Double,z:Double,model:String,device:String,gt:String)
  case class phoneGyroCoordinates(index:Int,user:String,x:Double,y:Double,z:Double,model:String,device:String,gt:String)
  case class phoneAccGyroCoordinates(index:Int,user:String, x_a:Double, y_a:Double, z_a:Double, x_g:Double, y_g:Double, z_g:Double, gt:String)
  // Creation_Time, Model, Index, x, Device, z, gt, User, y, Arrival_Time


  def main(args: Array[String]): Unit = {

    import spark.implicits._

    val phoneAccCsv = "hdfs://h1:9000/user/hduser/activity/Phones_accelerometer.csv.gz"
    val phoneGyroCsv = "hdfs://h1:9000/user/hduser/activity/Phones_gyroscope.csv.gz"

    val phoneAccParquet = "hdfs://h1:9000/user/hduser/activity/Phones_accelerometer.parquet"
    val phoneGyroParquet = "hdfs://h1:9000/user/hduser/activity/Phones_gyroscope.parquet"

    //  convertToParquet(Array(phoneAccCsv, phoneGyroCsv), Array(phoneAccParquet, phoneGyroParquet))

    etlWithParquetToDataset(Array(phoneAccParquet, phoneGyroParquet))

    //  val ds = etlWithParquet(Array(phoneAccParquet, phoneGyroParquet))

    // val sampleDs=ds.sample(false,0.02)

    //  trainRandomForestModel(sampleDs)

    close

  }


  def convertToParquet(sourcePaths: Array[String], destinationPaths: Array[String]): Unit = {

    val phoneAccRaw = spark.read.option("header", "true").option("inferSchema", "true").csv(sourcePaths(0)).coalesce(32)
    val phoneGyroRaw = spark.read.option("header", "true").option("inferSchema", "true").csv(sourcePaths(1)).coalesce(32)

    phoneAccRaw.write.parquet(destinationPaths(0))
    phoneGyroRaw.write.parquet(destinationPaths(1))

    println("I converted the files to Parquet !!!")

  }


  def etlWithParquetToDataset(sourcePaths: Array[String]): Unit = {
    sqlContext.setConf("spark.sql.shuffle.partitions", "32")

    val phoneAcc = spark.read.option("header", "true").option("inferSchema", "true").parquet(sourcePaths(0))
    val phoneGyro = spark.read.option("header", "true").option("inferSchema", "true").parquet(sourcePaths(1))

    import spark.implicits._

    val phoneAccDS= phoneAcc.drop('Creation_Time).drop('Arrival_Time).as[phoneAccCoordinates]
    val phoneGyroDS= phoneGyro.drop('Creation_Time).drop('Arrival_Time).as[phoneGyroCoordinates]

    phoneAccDS.printSchema()
    phoneGyroDS.printSchema()

    /*
    val phoneAccGyro = phoneAccDS.joinWith(phoneGyroDS,
      (phoneAccDS.col("Index") === phoneGyroDS.col("Index"))
        && (phoneAccDS.col("User") === phoneGyroDS.col("User"))
        && (phoneAccDS.col("Model") === phoneGyroDS.col("Model"))
        && (phoneAccDS.col("Device") === phoneGyroDS.col("Device")),"inner" )
    */
    /*
    xs.as("xs").joinWith(
  ys.as("ys"), ($"xs._1" === $"ys._1") && ($"xs._2" === $"ys._2"), "left")
     */


    val phoneAccGyro = phoneAccDS.as("ACC").joinWith(phoneGyroDS.as("GYRO"),
      ($"ACC.Index"=== $"GYRO.Index")&& ($"ACC.User"=== $"GYRO.User") && ($"ACC.Model"=== $"GYRO.Model") && ($"ACC.Device"=== $"GYRO.Device"), "inner")

    println(phoneAccGyro.count())
    phoneAccGyro.show(100)

    //10,835,775

    //
    //  phoneAccDS.printSchema()

    //  val phoneGyroDS= phoneGyro.as[phoneGyroCoordinates]
    //  phoneGyroDS.printSchema()


    /*
    root
 |-- Index: integer (nullable = true)
 |-- Arrival_Time: long (nullable = true)
 |-- Creation_Time: long (nullable = true)
 |-- x: double (nullable = true)
 |-- y: double (nullable = true)
 |-- z: double (nullable = true)
 |-- User: string (nullable = true)
 |-- Model: string (nullable = true)
 |-- Device: string (nullable = true)
 |-- gt: string (nullable = true)

root
 |-- Index: integer (nullable = true)
 |-- Arrival_Time: long (nullable = true)
 |-- Creation_Time: long (nullable = true)
 |-- x: double (nullable = true)
 |-- y: double (nullable = true)
 |-- z: double (nullable = true)
 |-- User: string (nullable = true)
 |-- Model: string (nullable = true)
 |-- Device: string (nullable = true)
 |-- gt: string (nullable = true)


 +-----+----+------+--------+-------------------+------------------+-----------------+-----+--------------------+--------------------+--------------------+-----+--------+--------+
|Index|User| Model|  Device|                x_a|               y_a|              z_a| gt_a|                 x_g|                 y_g|                 z_g| gt_g|gt_combi|gt_grand|
+-----+----+------+--------+-------------------+------------------+-----------------+-----+--------------------+--------------------+--------------------+-----+--------+--------+
|    0|   h|    s3|    s3_1|           1.436521|         0.8714894|         9.519346|  sit|        -0.010995574|         0.011301007|        0.0021380284|  sit|    true|     sit|
|    1|   e|    s3|    s3_1|         -2.3750482|       -0.08619126|           9.5385|stand|         0.022907447|         -0.01740966|           -0.086132|stand|    true|   stand|
|    2|   c|    s3|    s3_1|          -2.873042|       -0.92895025|         9.356541|stand|         0.006414085|         0.002443461|         0.004581489|stand|    true|   stand|
|    2|   d|nexus4|nexus4_1|-2.1079407000000003|         -1.188858|         9.999176|stand|         0.016433716|        0.0028381348|        -0.014266968|stand|    true|   stand|
|    2|   i|    s3|    s3_1|        -0.40222588|       -0.21068975|         9.797073|stand|9.162978999999999E-4|         0.007330383|-0.00885754599999...|stand|    true|   stand|
|    3|   b|s3mini|s3mini_1|-3.3470940000000002|1.8423381999999997|9.327810000000001|stand|-0.01415079100000...|0.029765457000000002|  0.6643551999999999|stand|    true|   stand|
|    4|   g|nexus4|nexus4_1|         -2.1472168|1.7068633999999998|9.859924000000001|stand|3.356933600000000...|          0.02720642|         0.010025024|stand|    true|   stand|
|    5|   a|s3mini|s3mini_1| 5.8215012999999995|        -1.1935096|        7.4567413|stand|-0.00536754170000...|         0.005123562|        0.0024397916|stand|    true|   stand|
|    5|   c|    s3|    s3_1|         -2.8443117|        -0.8331822|         9.337387|stand|         0.011301007|0.001832595799999...|         0.003970624|stand|    true|   stand|
|    5|   c|    s3|    s3_2|         -3.0358477|        -0.7374141|         8.963891|stand|         -0.01863139|         0.012217305|          0.05375614|stand|    true|   stand|
+-----+----+------+--------+-------------------+------------------+-----------------+-----+--------------------+--------------------+--------------------+-----+--------+--------+
only showing top 10 rows

root
 |-- Index: integer (nullable = true)
 |-- User: string (nullable = true)
 |-- Model: string (nullable = true)
 |-- Device: string (nullable = true)
 |-- x_a: double (nullable = true)
 |-- y_a: double (nullable = true)
 |-- z_a: double (nullable = true)
 |-- gt_a: string (nullable = true)
 |-- x_g: double (nullable = true)
 |-- y_g: double (nullable = true)
 |-- z_g: double (nullable = true)
 |-- gt_g: string (nullable = true)
 |-- gt_combi: boolean (nullable = true)
 |-- gt_grand: string (nullable = true)

     */





  }

  def etlWithParquet(sourcePaths: Array[String]): DataFrame = {

    spark.sqlContext.setConf("spark.sql.shuffle.partitions", "32")

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
    phoneAccGyroCleaned.show(10)

    phoneAccGyroCleaned.printSchema()
    phoneAccGyroCleaned

  }

  def trainRandomForestModel(data: DataFrame): Unit = {

    val seed = 5043
    val metric = "accuracy"
    val depth = 30
    val trees = 50
    val bin = 100

    println("Number of records  before removing NA:"+data.count())
    println("Number of records after removing NA :"+data.na.drop.count())

    val Array(trainingData, testData) = data.na.drop.randomSplit(Array(0.7, 0.3), seed)

    trainingData.cache()
    testData.cache()

    println("Number of records used for the ML trainingData stage :"+trainingData.count())

    val featureCols = data.select("x_a", "y_a", "z_a", "x_g", "y_g", "z_g").columns

    val labelIndexer = new StringIndexer().setInputCol("gt_grand").setOutputCol("indexed_gt_grand").fit(data)

    // Scale

    println("labelIndexer :")

    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("indexed_features")

    // val assembler = new VectorAssembler().setInputCols(sScaler).setOutputCol("indexed_features")

    println("assembler :")

    val classifier = new RandomForestClassifier()
      .setImpurity("gini")
      .setMaxDepth(depth)
      .setNumTrees(trees)
      .setFeatureSubsetStrategy("auto")
      .setSeed(seed)
      .setLabelCol("indexed_gt_grand")
      .setFeaturesCol("indexed_features").setMaxBins(bin)

    println("classifier :")

    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predicted_label")
      .setLabels(labelIndexer.labels)

    println("labelConverter :")

    val pipeline = new Pipeline().setStages(Array(labelIndexer, assembler, classifier, labelConverter))

    println("pipeline :")

    val model = pipeline.fit(trainingData)

    println("model :")

    val predictions = model.transform(testData)

    println("predictions :")

    predictions.select("predicted_label", "gt_grand", "indexed_features").show(10)

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexed_gt_grand")
      .setPredictionCol("prediction")
      .setMetricName(metric)

    println("Evaluator Islargerbetter "+evaluator.isLargerBetter)


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


    val paramGrid = new ParamGridBuilder()
      .addGrid(classifier.maxBins, Array(40,150))
      .addGrid(classifier.maxDepth, Array(3,30))
      .addGrid(classifier.numTrees, Array(40,120))
      .addGrid(classifier.impurity, Array("entropy", "gini"))
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(10)

    val pipelineFittedModel = cv.fit(trainingData)

    val predictions2 = pipelineFittedModel.transform(testData)
    val accuracy2 = evaluator.evaluate(predictions2)

    println("Test Error after hyper-parameter tuning = " + (1.0 - accuracy2))

    println("Best model " + pipelineFittedModel.bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel].params.toString)


    val paramMap=pipelineFittedModel
      .bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel]
      .stages(0)
      .extractParamMap

    val bestModel=pipelineFittedModel
      .bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel]

    val rm2 = new RegressionMetrics(
      predictions2.select("prediction", "indexed_gt_grand").rdd.map(x =>
        (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double]))
    )

    println("MSE: " + rm2.meanSquaredError)
    println("MAE: " + rm2.meanAbsoluteError)
    println("RMSE Squared: " + rm2.rootMeanSquaredError)
    println("R Squared: " + rm2.r2)
    println("Explained Variance: " + rm2.explainedVariance + "\n")


  }




}