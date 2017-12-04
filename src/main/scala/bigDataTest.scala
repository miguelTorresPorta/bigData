import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Row

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.Pipeline

object bigDataTest {

  def main (args: Array[String]){

    // Build SparkSession Object
    val spark = SparkSession
      .builder
      .master("local")
      .appName("spark")
      .enableHiveSupport
      .getOrCreate

    import spark.implicits._

    // Input file
    //val inputFilePath = "/home/jelguero/spark-project/2008.csv" // Jaime
    //val inputFilePath = "" // Miguel
    val inputFilePath = "/home/jelguero/spark-project/2008_1M.csv" // Jaime - Working file (1M rows)

    // Load file
    var df = spark
      .read
      .format("csv")
      .option("header", "true")
      .load(inputFilePath)

    // Save smaller file to work (only once)
    //df.limit(1000000).write.format("csv").save("/home/jelguero/spark-project/2008_1M.csv")

    // Columns (29)

    // Year         ArrTime             CRSElapsedTime    Distance            CarrierDelay
    // Month        CRSArrTime          AirTime           TaxiIn              WeatherDelay
    // DayofMonth   UniqueCarrier       ArrDelay          TaxiOut             NASDelay
    // DayOfWeek    FlightNum           DepDelay          Cancelled           SecurityDelay
    // DepTime      TailNum             Origin            CancellationCode    LateAircraftDelay
    // CRSDepTime   ActualElapsedTime   Dest              Diverted

    // Remove forbidden columns
    df = df.drop("ArrTime","ActualElapsedTime","AirTime","TaxiIn","Diverted","CarrierDelay","WeatherDelay","NASDelay",
      "SecurityDelay","LateAircraftDelay")

    // Remove non-used columns
    df = df.drop("Year","FlightNum","TaxiOut","Cancelled","CancellationCode")

    // Adjust data types
    df = df.select(
      df.col("Month").cast("int"),
      df.col("DayofMonth").cast("int"),
      df.col("DayOfWeek").cast("int"),
      df.col("DepTime").cast("int"),
      df.col("CRSDepTime").cast("int"),
      df.col("CRSArrtime").cast("int"),
      df.col("UniqueCarrier"),
      //df.col("TailNum"),
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

    // Remove rows with wrong values

    /* (This part is commented because there are no wrong values)

    // Month - 0 rows removed
    val Month = 1 to 12
    df = df.filter(df.col("Month").isin(Month : _*))

    // Day of Month - 0 rows removed
    val DayofMonth = 1 to 31
    df = df.filter(df.col("DayofMonth").isin(DayofMonth : _*))

    // Day of Week - 0 rows removed
    val DayOfWeek = 1 to 7
    df = df.filter(df.col("DayOfWeek").isin(DayOfWeek : _*))

    // Departure Time - 0 rows removed
    df.select(max("DepTime")).show // DepTime Max = 2400
    df.select(min("DepTime")).show // DepTime Min = 1
    var DepTime  = 1 to 59 toList;
    for (h <- 1 to 23; m <- 0 to 59){
      var time = h * 100 + m
      DepTime :+= time
    }
    DepTime :+= 2400
    df = df.filter(col("DepTime").isin(DepTime : _*))

    // [...]

    // Cancelled
    df.select(max("Cancelled")).show // Cancelled Max = 0
    df.select(min("Cancelled")).show // Cancelled Min = 0
    // There are no cancelled flights (they probably were removed when removing NAs) -> This column should be removed

    */

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // DATA TRANSFORMATIONS

    // Transform categorical variables into nominal

    // UniqueCarrier

    val all_UniqueCarrier = df.select("UniqueCarrier").distinct.rdd.map((r: Row) => r.getString(0)).collect
    var Map_UniqueCarrier = scala.collection.mutable.Map[String, Int]()
    var deMap_UniqueCarrier = scala.collection.mutable.Map[Int, String]()
    var index = 0
    for (i <- 0 until all_UniqueCarrier.length) {
      Map_UniqueCarrier += (all_UniqueCarrier(i) -> index)
      deMap_UniqueCarrier += (index -> all_UniqueCarrier(i))
      index += 1
    }

    def conv_UniqueCarrier (name : String) : Int = {
      var code = Map_UniqueCarrier(name)
      return code
    }
    var conv_UniqueCarrier_udf = udf(conv_UniqueCarrier _)

    def deconv_UniqueCarrier (code : Int) : String = {
      var name = deMap_UniqueCarrier(code)
      return name
    }
    var deconv_UniqueCarrier_udf = udf(deconv_UniqueCarrier _)

    df = df.withColumn("UniqueCarrier", conv_UniqueCarrier_udf(col("UniqueCarrier")))

    // TailNum

    /*
    val all_TailNum = df.select("TailNum").distinct.rdd.map((r: Row) => r.getString(0)).collect
    var Map_TailNum = scala.collection.mutable.Map[String, Int]()
    var deMap_TailNum = scala.collection.mutable.Map[Int, String]()
    index = 0
    for (i <- 0 until all_TailNum.length) {
      Map_TailNum += (all_TailNum(i) -> index)
      deMap_TailNum += (index -> all_TailNum(i))
      index += 1
    }

    def conv_TailNum (name : String) : Int = {
      var code = Map_TailNum(name)
      return code
    }
    val conv_TailNum_udf = udf(conv_TailNum _)

    def deconv_TailNum(code : Int) : String = {
      var name = deMap_TailNum(code)
      return name
    }
    val deconv_TailNum_udf = udf(deconv_TailNum _)

    df = df.withColumn("TailNum", conv_TailNum_udf(col("TailNum")))
    */

    // Origin

    val all_Origin = df.select("Origin").distinct.rdd.map((r: Row) => r.getString(0)).collect
    var Map_Origin = scala.collection.mutable.Map[String, Int]()
    var deMap_Origin = scala.collection.mutable.Map[Int, String]()
    index = 0
    for (i <- 0 until all_Origin.length) {
      Map_Origin += (all_Origin(i) -> index)
      deMap_Origin += (index -> all_Origin(i))
      index += 1
    }

    def conv_Origin (name : String) : Int = {
      var code = Map_Origin(name)
      return code
    }
    var conv_Origin_udf = udf(conv_Origin _)

    def deconv_Origin (code : Int) : String = {
      var name = deMap_Origin(code)
      return name
    }
    var deconv_Origin_udf = udf(deconv_Origin _)

    df = df.withColumn("Origin", conv_Origin_udf(col("Origin")))

    // Dest

    val all_Dest = df.select("Dest").distinct.rdd.map((r: Row) => r.getString(0)).collect
    var Map_Dest = scala.collection.mutable.Map[String, Int]()
    var deMap_Dest = scala.collection.mutable.Map[Int, String]()
    index = 0
    for (i <- 0 until all_Dest.length) {
      Map_Dest += (all_Dest(i) -> index)
      deMap_Dest += (index -> all_Dest(i))
      index += 1
    }

    def conv_Dest (name : String) : Int = {
      var code = Map_Dest(name)
      return code
    }
    var conv_Dest_udf = udf(conv_Dest _)

    def deconv_Dest (code : Int) : String = {
      var name = deMap_Dest(code)
      return name
    }
    var deconv_Dest_udf = udf(deconv_Dest _)

    df = df.withColumn("Dest", conv_Dest_udf(col("Dest")))


    // Transform time from hhmm format into total minutes

    def hhmmToMinutes (hhmm : Int) : Int = {
      var minutes = hhmm
      if (hhmm > 60) {
        minutes = (hhmm / 100) * 60 + (hhmm % 100)
      }
      return minutes
    }
    var hhmmToMinutes_udf =udf(hhmmToMinutes _)

    def MinutesToHHMM (Minutes : Int) : Int = {
      var HHMM = Minutes
      if (Minutes > 60) {
        HHMM = (Minutes / 60) * 100 + (Minutes % 60)
      }
      return HHMM
    }
    var MinutesToHHMM_udf = udf(MinutesToHHMM _)

    df = df.withColumn("DepTime", hhmmToMinutes_udf(col("DepTime")))
    df = df.withColumn("CRSDepTime", hhmmToMinutes_udf(col("CRSDepTime")))
    df = df.withColumn("CRSArrtime", hhmmToMinutes_udf(col("CRSArrtime")))


    // Create new column "CRSArrTimeOrigin" = "CRSDepTime" + "CRSElapsedTime"
    df = df.withColumn("CRSArrTimeOrigin", col("CRSDepTime") + col("CRSElapsedTime"))

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////                                           TEST AREA                                           //////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    var df2 = df.select(
      df.col("Month"),
      df.col("DayofMonth"),
      df.col("DayOfWeek"),
      df.col("CRSDepTime"),
      df.col("CRSArrTime"),
      df.col("UniqueCarrier"),
      df.col("CRSElapsedTime"),
      df.col("DepDelay"),
      df.col("Origin"),
      df.col("Dest"),
      df.col("Distance"),
      df.col("ArrDelay")
    )

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
        "UniqueCarrier",
        "CRSElapsedTime",
        "DepDelay",
        "Origin",
        "Dest",
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
    model.transform(test).show(truncate = false)

    var df3 = model.transform(test)

    df3 = df3.withColumn("error", abs(col("ArrDelay") - col("prediction")))

    df3.select(avg(col("error"))).show()
    df3.select(max(col("error"))).show()
    df3.select(min(col("error"))).show()

    df3.show()

    //    val lrModel = lr.fit(output)
    //
    //    println(s"Coefficients: ${lrModel.coefficients}")
    //    println(s"Intercept: ${lrModel.intercept}")
    //
    //    val trainingSummary = lrModel.summary
    //    println(s"numIterations: ${trainingSummary.totalIterations}")
    //    println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
    //    trainingSummary.residuals.show()
    //    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    //    println(s"r2: ${trainingSummary.r2}")

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////                                              END                                              //////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  }


}
