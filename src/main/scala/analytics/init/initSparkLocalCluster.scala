package analytics.init

package analytics.init

import com.typesafe.config.ConfigFactory
import org.apache.log4j.{Level, LogManager, Logger}
import org.apache.spark.sql.SparkSession

trait InitSparkLocalCluster {

  val spark = SparkSession.builder()
    .appName("ActivityRecognition")
    .master("spark://h1:7077")
    .config("spark.jars", "/home/hduser/SparkExamples/target/scala-2.11/classes/advanced.jar")
    .config("spark.driver.memory", "8g")
    .config("spark.executor.memory", "24g")
    .getOrCreate()

  val sc = spark.sparkContext
  val sqlContext = spark.sqlContext

  val logger = LogManager.getRootLogger
  logger.setLevel(Level.INFO)

  val conf =  ConfigFactory.load()

  def close = {
    spark.close()
  }

}
