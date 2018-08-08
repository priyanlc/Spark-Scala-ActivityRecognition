# _Spark-Activity-recognition POC
A project with scala, apache spark built using gradle.

## Prerequisites
- [Java](https://java.com/en/download/)
- [Gradle](https://gradle.org/)
- [Scala](https://www.scala-lang.org/)

## Build and Demo process

### Clone the Repo
`git clone https://github.com/priyanlc/Spark-Scala-ActivityRecognition.git`

### Build
`./gradlew clean build`
### Run
`./gradlew run`
### All Together
`./gradlew clean run`
### Build Uber Jar to deploy in Spark cluster
./gradlew shadowJar

## to run as a local spark process do the following changes in InitSparkCluster
``` val spark = SparkSession.builder()
    .appName("ActivityRecognition")
    .master("local[*]")
    .getOrCreate()
```

## to Submit to Spark cluster
spark-submit \
    --class analytics.activityrecognition.ActivityRecognition \
    --master spark://h1:7077 \
    --deploy-mode client \
    --executor-memory 28G \
    --driver-memory 6G \
    --total-executor-cores 32 \
     /home/hduser/jars/spark-activity-recognition-1.0-SNAPSHOT-shadow.jar --to-parquet false --hyper-param true


## Using this Repo
Just import it into your favorite IDE as a gradle project. Tested with IntelliJ to work. Or use your favorite editor and build from command line with gradle.

## Libraries Included
- Spark - 2.2.0

## Useful Links
- [Spark Docs - Root Page](http://spark.apache.org/docs/latest/)
- [Spark Programming Guide](http://spark.apache.org/docs/latest/programming-guide.html)
- [Spark Latest API docs](http://spark.apache.org/docs/latest/api/)
- [Scala API Docs](http://www.scala-lang.org/api/2.12.1/scala/)
 
## Issues or Suggestions
- Raise one on github
- Send me a mail -> priyanchandrapala @ yahoo dot co dot uk (Remove the spaces and dot = .)


