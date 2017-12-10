import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.{Column, DataFrame}

object Tools {


  def hhmmToMinutes (hhmm : Column) : Column = {
    val hh = (hhmm/100) - ((hhmm/100)%1)
    val min = (hh) * 60 + (hhmm % 100)

    return min
  }

  def indexer (columnNameInput: String, columnNameOutput: String, df: DataFrame) : DataFrame = {

    val indexer = new StringIndexer()
      .setInputCol(columnNameInput)
      .setOutputCol(columnNameOutput)

    return indexer.fit(df).transform(df)
  }

  def joinColumns(a: Column, b: Column): Column = {
    return a + ("") + b

  }


}
