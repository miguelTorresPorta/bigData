import org.apache.spark.sql.{Column, Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.log4j.Logger
import org.apache.log4j.Level

object bigDataTest {

  def main (args: Array[String]){

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val startTime = System.currentTimeMillis()

    if(args.length == 0) {
      println("Problem reading the file! Please insert a good path for the input file")
      throw new IllegalArgumentException("Not input parameter")
    }


    // Build SparkSession Object
    val spark = SparkSession
      .builder
      .master("local")
      .appName("spark")
      .enableHiveSupport
      .getOrCreate


    val inputFilePath = args(0)
    println("Reading " + inputFilePath + " file")

    // Load file
    var df = spark
      .read
      .format("csv")
      .option("header", "true")
      .load(inputFilePath)

    // Columns (29)

    // Year         ArrTime             CRSElapsedTime    Distance            CarrierDelay
    // Month        CRSArrTime          AirTime           TaxiIn              WeatherDelay
    // DayofMonth   UniqueCarrier       ArrDelay          TaxiOut             NASDelay
    // DayOfWeek    FlightNum           DepDelay          Cancelled           SecurityDelay
    // DepTime      TailNum             Origin            CancellationCode    LateAircraftDelay
    // CRSDepTime   ActualElapsedTime   Dest              Diverted


    // Adjust data types
    df = df.select(
      df.col("Month").cast("int"),
      df.col("DayofMonth").cast("int"),
      df.col("DayOfWeek").cast("int"),
      df.col("DepTime").cast("int"),
      df.col("CRSDepTime").cast("int"),
      df.col("CRSArrtime").cast("int"),
      df.col("UniqueCarrier"),
      df.col("CRSElapsedTime").cast("int"),
      df.col("ArrDelay").cast("int"),
      df.col("DepDelay").cast("int"),
      df.col("Origin"),
      df.col("Dest"),
      df.col("Distance").cast("int")
    )

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // DATA CLEANSING

    // Remove rows containing NAs - 154,704 rows removed
    df = df.na.drop

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // DATA TRANSFORMATIONS

    // Transform categorical variables into nominal

    // UniqueCarrier
    df = Tools.indexer("UniqueCarrier", "UniqueCarrierIndex", df)

    // Origin
    df = Tools.indexer("Origin", "OriginIndex", df)

    // Dest
    df = Tools.indexer("Dest", "DestIndex", df)

    // Transform time from hhmm format into total minutes
    df = df.withColumn("DepTime", Tools.hhmmToMinutes(col("DepTime")))
      .withColumn("CRSDepTime", Tools.hhmmToMinutes(col("CRSDepTime")))
      .withColumn("CRSArrtime", Tools.hhmmToMinutes(col("CRSArrtime")))
      .withColumn("CRSArrTimeOrigin", col("CRSDepTime") + col("CRSElapsedTime"))

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // MODEL CREATION

    val df2 = df.select(
      df.col("Month"),
      df.col("DayofMonth"),
      df.col("DayOfWeek"),
      df.col("CRSDepTime"),
      df.col("CRSArrTime"),
      df.col("UniqueCarrierIndex"),
      df.col("CRSElapsedTime"),
      df.col("DepDelay"),
      df.col("OriginIndex"),
      df.col("DestIndex"),
      df.col("Distance"),
      df.col("ArrDelay")
    )

    df2.show()

    val split = df2.randomSplit(Array(0.7, 0.3))
    val training = split(0)
    val test = split(1)

    val assembler = new VectorAssembler()
      .setInputCols(Array(
        "Month",
        "DayofMonth",
        "DayOfWeek",
        "CRSDepTime",
        "CRSArrTime",
        "UniqueCarrierIndex",
        "CRSElapsedTime",
        "DepDelay",
        "OriginIndex",
        "DestIndex",
        "Distance"))
      .setOutputCol("features")

    val output = assembler.transform(df2)

    var lr = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("ArrDelay")
      .setMaxIter(10)
      .setElasticNetParam(0.8)

    val pipeline = new Pipeline()
      .setStages(Array(assembler, lr))

    val model = pipeline.fit(training)
    model.transform(test)//.show(truncate = false)

    var df3 = model.transform(test)

    df3.show()
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // MODEL EVALUATION

    // ERROR = ArrDelay - prediction
    df3 = df3.withColumn("error", abs(col("ArrDelay") - col("prediction")))

    df3.select(avg(col("error"))).show() // Avg. Error = 9.255627202272636 minutes
    df3.select(max(col("error"))).show() // Max. Error = 1038.015245074952 minutes
    df3.select(min(col("error"))).show() // Min. Error = 9.232469142972377E-6 minutes

    val evaluator = new RegressionEvaluator()
      .setLabelCol("ArrDelay")
      .setPredictionCol("prediction")

    // Root Mean Square Error
    val rmse = evaluator.setMetricName("rmse").evaluate(df3) // rmse = 13.913056017643722
    println("\nRoot Mean Square Error (RMSE) = " + rmse)

    // Coefficient of Determination
    val r2 = evaluator.setMetricName("r2").evaluate(df3) // r2 = 0.869393116810725
    println("\nr2 = " + r2)

    val endTime = System.currentTimeMillis()
    val totalTime = endTime - startTime
    println("Total execution time " + (totalTime/1000)/60 + " minutes" )

  }




}