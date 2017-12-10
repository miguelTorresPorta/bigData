import org.apache.spark.sql.SparkSession

import org.apache.log4j.Logger
import org.apache.log4j.Level

object bigData {

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

    df = ModelOperations.dataCleansing(df)

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // DATA TRANSFORMATIONS

    // Put together Day of month and Month
    df = ModelOperations.dataTransformations(df)

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // MODEL CREATION

    val df2 = ModelOperations.model(df)

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // MODEL EVALUATION

    ModelOperations.modelEvaluation(df2)

    val endTime = System.currentTimeMillis()
    val totalTime = endTime - startTime
    println("Total execution time " + (totalTime/1000)/60 + " minutes" )

  }

}