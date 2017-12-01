import org.apache.spark.sql.SparkSession

object bigDataTest {

  def main (args: Array[String]){

    val sparkSession = SparkSession
      .builder
      .master("local")
      .appName("sparkSession")
      .enableHiveSupport()
      .getOrCreate()


    val df = sparkSession.read.option("header","true")
      .csv("file:/Users/migueltorresporta/Documents/UPM/BigData/2008.csv")

    df.printSchema()
    df.show()

    df.createGlobalTempView("Month")

    //sparkSession.sql()
  }


}
