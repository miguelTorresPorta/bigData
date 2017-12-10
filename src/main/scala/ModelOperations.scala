import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

object ModelOperations {



  /**
    * Method the clean a data frame
    * @param df
    * @return
    */
  def dataCleansing (df : DataFrame): DataFrame = {
    // DATA CLEANSING

    // Remove rows containing NAs - 154,704 rows removed
    var newDf = df.na.drop

    // Month - 0 rows removed
    val Month = 1 to 12
    newDf = newDf.filter(newDf.col("Month").isin(Month : _*))
    // Day of Month - 0 rows removed
    val DayofMonth = 1 to 31
    newDf = newDf.filter(newDf.col("DayofMonth").isin(DayofMonth : _*))
    // Day of Week - 0 rows removed
    val DayOfWeek = 1 to 7
    newDf = newDf.filter(newDf.col("DayOfWeek").isin(DayOfWeek : _*))
    // Departure Time - 0 rows removed
    newDf.select(max("DepTime")).show // DepTime Max = 2400
    newDf.select(min("DepTime")).show // DepTime Min = 1
    var DepTime  = 1 to 59 toList;
    for (h <- 1 to 23; m <- 0 to 59){
      var time = h * 100 + m
      DepTime :+= time
    }
    DepTime :+= 2400
    newDf = newDf.filter(col("DepTime").isin(DepTime : _*))

    newDf

  }

  def dataTransformations(df : DataFrame) : DataFrame = {

    // Put together Day of month and Month

    var newDf = df.withColumn("DayMonth", concat(col("DayOfMonth"), col("Month")))
    newDf = Tools.indexer("DayMonth", "DayMonthIndex", newDf)

    // Transform categorical variables into nominal

    // UniqueCarrier
    newDf = Tools.indexer("UniqueCarrier", "UniqueCarrierIndex", newDf)

    // Origin
    newDf = Tools.indexer("Origin", "OriginIndex", newDf)

    // Dest
    newDf = Tools.indexer("Dest", "DestIndex", newDf)

    // Transform time from hhmm format into total minutes
    newDf = newDf.withColumn("DepTime", Tools.hhmmToMinutes(col("DepTime")))
      .withColumn("CRSDepTime", Tools.hhmmToMinutes(col("CRSDepTime")))
      .withColumn("CRSArrtime", Tools.hhmmToMinutes(col("CRSArrtime")))
      .withColumn("CRSArrTimeOrigin", col("CRSDepTime") + col("CRSElapsedTime"))

    newDf.show()

    newDf
  }

  def model (df : DataFrame) : DataFrame = {

    val df2 = df.select(
      df.col("Month"),
      df.col("DayMonthIndex"),
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
        "DayMonthIndex",
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

    val lr = new LinearRegression()
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

    df3
  }

  def modelEvaluation (df: DataFrame)  {

    // ERROR = ArrDelay - prediction
    var df3 = df.withColumn("error", abs(col("ArrDelay") - col("prediction")))

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


  }

}
