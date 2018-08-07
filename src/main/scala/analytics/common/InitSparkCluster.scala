package analytics.common

import org.apache.log4j.{Level, LogManager, Logger}
import org.apache.spark.sql.SparkSession

trait InitSparkCluster {

  val spark = SparkSession.builder()
    .appName("ActivityRecognition")
    .master("spark://h1:7077")
    .config("spark.jars", "/home/hduser/SparkExamples/target/scala-2.11/classes/advanced.jar")
    .config("spark.driver.memory", "6g")
    .config("spark.executor.memory", "30g")
    .getOrCreate()

  val sc = spark.sparkContext
  val sqlContext = spark.sqlContext

  private def init = {
    sc.setLogLevel("WARN")
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)
    LogManager.getRootLogger.setLevel(Level.WARN)
  }
  init

  def close = {
    spark.close()
  }

}
