package analytics.init

import com.typesafe.config.ConfigFactory
import org.apache.log4j.{Level, LogManager, Logger}
import org.apache.spark.sql.SparkSession

trait InitSparkCluster {

  val spark = SparkSession.builder()
    .appName("ActivityRecognition")
    .getOrCreate()

  // for local cluster add this line to the Session builder
  // .master("local[*]")

  val sc = spark.sparkContext
  val sqlContext = spark.sqlContext

  val logger = LogManager.getRootLogger
  logger.setLevel(Level.WARN)

  val conf =  ConfigFactory.load()

  def close = {
    spark.close()
  }

}
