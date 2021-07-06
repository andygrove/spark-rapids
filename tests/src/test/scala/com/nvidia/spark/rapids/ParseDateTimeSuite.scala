/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark.rapids

import java.sql.{Date, Timestamp}
import scala.collection.mutable.ListBuffer
import org.scalatest.BeforeAndAfterEach
import org.apache.spark.SparkConf
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.functions.{col, to_date, to_timestamp, unix_timestamp}
import org.apache.spark.sql.internal.SQLConf

import scala.util.Random

class ParseDateTimeSuite extends SparkQueryCompareTestSuite with BeforeAndAfterEach {

  override def beforeEach() {
    GpuOverrides.removeAllListeners()
  }

  override def afterEach() {
    GpuOverrides.removeAllListeners()
  }

  testSparkResultsAreEqual("to_date dd/MM/yy (fall back)",
    datesAsStrings,
    conf = new SparkConf().set(SQLConf.LEGACY_TIME_PARSER_POLICY.key, "CORRECTED")
        .set(RapidsConf.INCOMPATIBLE_DATE_FORMATS.key, "true")
        // until we fix https://github.com/NVIDIA/spark-rapids/issues/2118 we need to fall
        // back to CPU when parsing two-digit years
        .set(RapidsConf.TEST_ALLOWED_NONGPU.key,
          "ProjectExec,Alias,Cast,GetTimestamp,UnixTimestamp,Literal,ShuffleExchangeExec")) {
    df => df.withColumn("c1", to_date(col("c0"), "dd/MM/yy"))
  }

  testSparkResultsAreEqual("to_date yyyy-MM-dd",
      datesAsStrings,
      conf = new SparkConf().set(SQLConf.LEGACY_TIME_PARSER_POLICY.key, "CORRECTED")) {
    df => df.withColumn("c1", to_date(col("c0"), "yyyy-MM-dd"))
  }

  /*
,null], [1999-12-31 ,1999-12-31], [1999-12-31 11,1999-12-31], [1999-12-31 11:,1999-12-31], [1999-12-31 11:5,1999-12-31], [1999-12-31 11:59,1999-12-31], [1999-12-31 11:59:,1999-12-31], [1999-12-31 11:59:5,1999-12-31], [1999-12-31 11:59:59,1999-12-31], [  1999-12-31 11:59:59,1999-12-31], [	1999-12-31 11:59:59,1999-12-31], [	1999-12-31 11:59:59
,1999-12-31], [1999-12-31 11:59:59.,1999-12-31], [1999-12-31 11:59:59.9,1999-12-31], [ 1999-12-31 11:59:59.9,1999-12-31], [1999-12-31 11:59:59.99,1999-12-31], [1999-12-31 11:59:59.999,1999-12-31], [1999-12-31 11:59:59.9999,1999-12-31], [1999-12-31 11:59:59.99999,1999-12-31], [1999-12-31 11:59:59.999999,1999-12-31], [1999-12-31 11:59:59.9999999,1999-12-31], [1999-12-31 11:59:59.99999999,1999-12-31], [1999-12-31 11:59:59.999999999,1999-12-31], [31/12/1999,null], [31/12/1999 11:59:59.999,null],

,null], [1999-12-31 ,1999-12-31], [1999-12-31 11,1999-12-31], [1999-12-31 11:,1999-12-31], [1999-12-31 11:5,1999-12-31], [1999-12-31 11:59,1999-12-31], [1999-12-31 11:59:,1999-12-31], [1999-12-31 11:59:5,1999-12-31], [1999-12-31 11:59:59,1999-12-31], [  1999-12-31 11:59:59,1999-12-31], [	1999-12-31 11:59:59,1999-12-31], [	1999-12-31 11:59:59
,1999-12-31], [1999-12-31 11:59:59.,1999-12-31], [1999-12-31 11:59:59.9,1999-12-31], [ 1999-12-31 11:59:59.9,1999-12-31], [1999-12-31 11:59:59.99,1999-12-31], [1999-12-31 11:59:59.999,1999-12-31], [1999-12-31 11:59:59.9999,1999-12-31], [1999-12-31 11:59:59.99999,1999-12-31], [1999-12-31 11:59:59.999999,1999-12-31], [1999-12-31 11:59:59.9999999,1999-12-31], [1999-12-31 11:59:59.99999999,1999-12-31], [1999-12-31 11:59:59.999999999,1999-12-31], [31/12/1999,null], [31/12/1999 11:59:59.999,null],

[1999-12-3,1999-12-03], [1999-12-31,1999-12-31], [1999/12/31,null], [1999-12,null], [1999/12,null], [1975/06,null], [1975/06/18,null], [1975/06/18 06:48:57,null], [1999-12-29
[1999-12-3,null], [1999-12-31,1999-12-31], [1999/12/31,null], [1999-12,null], [1999/12,null], [1975/06,null], [1975/06/18,null], [1975/06/18 06:48:57,null], [1999-12-29
   */
  testSparkResultsAreEqual("to_date yyyy-MM-dd LEGACY",
    datesAsStrings,
    conf = new SparkConf().set(SQLConf.LEGACY_TIME_PARSER_POLICY.key, "LEGACY")) {
    df => df.withColumn("c1", to_date(col("c0"), "yyyy-MM-dd"))
  }

  testSparkResultsAreEqual("to_date dd/MM/yyyy",
      datesAsStrings,
      conf = new SparkConf().set(SQLConf.LEGACY_TIME_PARSER_POLICY.key, "CORRECTED")) {
    df => df.withColumn("c1", to_date(col("c0"), "dd/MM/yyyy"))
  }

  testSparkResultsAreEqual("to_date dd/MM/yyyy LEGACY",
    datesAsStrings,
    conf = new SparkConf().set(SQLConf.LEGACY_TIME_PARSER_POLICY.key, "LEGACY")) {
    df => df.withColumn("c1", to_date(col("c0"), "dd/MM/yyyy"))
  }

  testSparkResultsAreEqual("to_date parse date",
      dates,
      conf = new SparkConf().set(SQLConf.LEGACY_TIME_PARSER_POLICY.key, "CORRECTED")) {
    df => df.withColumn("c1", to_date(col("c0"), "yyyy-MM-dd"))
  }

  testSparkResultsAreEqual("to_date parse timestamp",
      timestamps,
      conf = new SparkConf().set(SQLConf.LEGACY_TIME_PARSER_POLICY.key, "CORRECTED")) {
    df => df.withColumn("c1", to_date(col("c0"), "yyyy-MM-dd"))
  }

  testSparkResultsAreEqual("to_timestamp yyyy-MM-dd",
      timestampsAsStrings,
      conf = new SparkConf().set(SQLConf.LEGACY_TIME_PARSER_POLICY.key, "CORRECTED")) {
    df => df.withColumn("c1", to_timestamp(col("c0"), "yyyy-MM-dd"))
  }

  testSparkResultsAreEqual("to_timestamp dd/MM/yyyy",
      timestampsAsStrings,
      conf = new SparkConf().set(SQLConf.LEGACY_TIME_PARSER_POLICY.key, "CORRECTED")) {
    df => df.withColumn("c1", to_timestamp(col("c0"), "dd/MM/yyyy"))
  }

  testSparkResultsAreEqual("to_date default pattern",
      datesAsStrings,
      new SparkConf().set(SQLConf.LEGACY_TIME_PARSER_POLICY.key, "CORRECTED")) {
    df => df.withColumn("c1", to_date(col("c0")))
  }

  testSparkResultsAreEqual("unix_timestamp parse date",
      timestampsAsStrings,
      new SparkConf().set(SQLConf.LEGACY_TIME_PARSER_POLICY.key, "CORRECTED")) {
    df => df.withColumn("c1", unix_timestamp(col("c0"), "yyyy-MM-dd"))
  }

  testSparkResultsAreEqual("unix_timestamp parse yyyy/MM",
    timestampsAsStrings,
    new SparkConf().set(SQLConf.LEGACY_TIME_PARSER_POLICY.key, "CORRECTED")) {
    df => df.withColumn("c1", unix_timestamp(col("c0"), "yyyy/MM"))
  }

  testSparkResultsAreEqual("to_unix_timestamp parse yyyy/MM",
    timestampsAsStrings,
    new SparkConf().set(SQLConf.LEGACY_TIME_PARSER_POLICY.key, "CORRECTED")) {
    df => {
      df.createOrReplaceTempView("df")
      df.sqlContext.sql("SELECT c0, to_unix_timestamp(c0, 'yyyy/MM') FROM df")
    }
  }

  testSparkResultsAreEqual("to_unix_timestamp parse yyyy/MM (improvedTimeOps)",
    timestampsAsStrings,
    new SparkConf().set(SQLConf.LEGACY_TIME_PARSER_POLICY.key, "CORRECTED")
        .set(RapidsConf.IMPROVED_TIMESTAMP_OPS.key, "true")) {
    df => {
      df.createOrReplaceTempView("df")
      df.sqlContext.sql("SELECT c0, to_unix_timestamp(c0, 'yyyy/MM') FROM df")
    }
  }

  testSparkResultsAreEqual("unix_timestamp parse timestamp",
      timestampsAsStrings,
      new SparkConf().set(SQLConf.LEGACY_TIME_PARSER_POLICY.key, "CORRECTED")) {
    df => df.withColumn("c1", unix_timestamp(col("c0"), "yyyy-MM-dd HH:mm:ss"))
  }

  testSparkResultsAreEqual("unix_timestamp parse timestamp LEGACY",
    timestampsAsStrings,
    new SparkConf().set(SQLConf.LEGACY_TIME_PARSER_POLICY.key, "LEGACY")) {
    df => df.withColumn("c1", unix_timestamp(col("c0"), "yyyy-MM-dd HH:mm:ss"))
  }

  testSparkResultsAreEqual("unix_timestamp parse timestamp millis (fall back to CPU)",
    timestampsAsStrings,
    new SparkConf().set(SQLConf.LEGACY_TIME_PARSER_POLICY.key, "CORRECTED")
      .set(RapidsConf.TEST_ALLOWED_NONGPU.key,
        "ProjectExec,Alias,UnixTimestamp,Literal,ShuffleExchangeExec")) {
    df => df.withColumn("c1", unix_timestamp(col("c0"), "yyyy-MM-dd HH:mm:ss.SSS"))
  }

  testSparkResultsAreEqual("unix_timestamp parse timestamp default pattern",
      timestampsAsStrings,
      new SparkConf().set(SQLConf.LEGACY_TIME_PARSER_POLICY.key, "CORRECTED")) {
    df => df.withColumn("c1", unix_timestamp(col("c0")))
  }

  test("fall back to CPU when policy is LEGACY") {
    val e = intercept[IllegalArgumentException] {
      val df = withGpuSparkSession(spark => {
        timestampsAsStrings(spark)
            .repartition(2)
            .withColumn("c1", unix_timestamp(col("c0"), "u"))
      }, new SparkConf().set(SQLConf.LEGACY_TIME_PARSER_POLICY.key, "LEGACY"))
      df.collect()
    }
    assert(e.getMessage.contains(
      "Part of the plan is not columnar class org.apache.spark.sql.execution.ProjectExec"))
  }

  test("unsupported format") {

    // capture plans
    val plans = new ListBuffer[SparkPlanMeta[SparkPlan]]()
    GpuOverrides.addListener(
        (plan: SparkPlanMeta[SparkPlan], _: SparkPlan, _: Seq[Optimization]) => {
      plans.append(plan)
    })

    val e = intercept[IllegalArgumentException] {
      val df = withGpuSparkSession(spark => {
        datesAsStrings(spark)
          .repartition(2)
          .withColumn("c1", to_date(col("c0"), "F"))
      }, new SparkConf().set(SQLConf.LEGACY_TIME_PARSER_POLICY.key, "CORRECTED"))
      df.collect()
    }
    assert(e.getMessage.contains(
      "Part of the plan is not columnar class org.apache.spark.sql.execution.ProjectExec"))

    val planStr = plans.last.toString
    assert(planStr.contains("Failed to convert Unsupported character: F"))
    // make sure we aren't suggesting enabling INCOMPATIBLE_DATE_FORMATS for something we
    // can never support
    assert(!planStr.contains(RapidsConf.INCOMPATIBLE_DATE_FORMATS.key))
  }

  test("parse now") {
    def now(spark: SparkSession) = {
      import spark.implicits._
      Seq("now").toDF("c0")
          .repartition(2)
          .withColumn("c1", unix_timestamp(col("c0"), "yyyy-MM-dd HH:mm:ss"))
    }
    val startTimeSeconds = System.currentTimeMillis()/1000L
    val cpuNowSeconds = withCpuSparkSession(now).collect().head.toSeq(1).asInstanceOf[Long]
    val gpuNowSeconds = withGpuSparkSession(now).collect().head.toSeq(1).asInstanceOf[Long]
    assert(cpuNowSeconds >= startTimeSeconds)
    assert(gpuNowSeconds >= startTimeSeconds)
    // CPU ran first so cannot have a greater value than the GPU run (but could be the same second)
    assert(cpuNowSeconds <= gpuNowSeconds)
  }

  // just show the failures so we don't have to manually parse all
  // the output to find which ones failed
  override def compareResults(
      sort: Boolean,
      maxFloatDiff: Double,
      fromCpu: Array[Row],
      fromGpu: Array[Row]): Unit = {
    assert(fromCpu.length === fromGpu.length)

    val failures = fromCpu.zip(fromGpu).zipWithIndex.filterNot {
      case ((cpu, gpu), _) => super.compare(cpu, gpu, 0.0001)
    }

    if (failures.nonEmpty) {
      val str = failures.map {
        case ((cpu, gpu), i) =>
          s"""
             |[#$i] CPU: $cpu
             |[#$i] GPU: $gpu
             |
             |""".stripMargin
      }.mkString("\n")
      fail(s"Mismatch between CPU and GPU for the following rows:\n$str")
    }
  }

  private def dates(spark: SparkSession) = {
    import spark.implicits._
    dateValues.toDF("c0")
  }

  private def timestamps(spark: SparkSession) = {
    import spark.implicits._
    tsValues.toDF("c0")
  }

  private def timestampsAsStrings(spark: SparkSession) = {
    import spark.implicits._
    timestampValues.toDF("c0")
  }

  private def datesAsStrings(spark: SparkSession) = {
    import spark.implicits._
    val values = Seq(
      DateUtils.EPOCH,
      DateUtils.NOW,
      DateUtils.TODAY,
      DateUtils.YESTERDAY,
      DateUtils.TOMORROW
    ) ++ timestampValues
    values.toDF("c0")
  }

  private def generateTimestampStrings(n: Int): Seq[String] = {
    val validChars = "0123456789:-. \t\n"
    val rand = new Random(0) // fixed seed
    val list = new ListBuffer[String]()
    for (_ <- 0 to n) {
      val len = rand.nextInt(32)
      val str = new StringBuilder(len)
      for (_ <- 0 to len) {
        str.append(validChars.charAt(rand.nextInt(validChars.length)))
      }
      list += str.toString
    }
    list
  }

  private val timestampValues = /*generateTimestampStrings(10000) ++*/ Seq(
    "",
    "null",
    null,
    "\n",
    "1999-1-1 ",
    "1999-1-1 11",
    "1999-1-1 11:",
    "1999-1-1 11:5",
    "1999-1-1 11:59",
    "1999-1-1 11:59:",
    "1999-1-1 11:59:5",
    "1999-1-1 11:59:59",
    "1999-1-1",
    "1999-1-1 ",
    "1999-1-1 1",
    "1999-1-1 1:",
    "1999-1-1 1:2",
    "1999-1-1 1:2:",
    "1999-1-1 1:2:3",
    "1999-1-1 1:2:3.",
    "1999-1-1 1:12:3.",
    "1999-1-1 11:12:3.",
    "1999-1-1 11:2:3.",
    "1999-1-1 11:2:13.",
    "1999-1-1 1:2:3.4",
    "1999-1-1 ",
    "1999-12-31 11",
    "1999-12-31 11:",
    "1999-12-31 11:5",
    "1999-12-31 11:59",
    "1999-12-31 11:59:",
    "1999-12-31 11:59:5",
    "1999-12-31 11:59:59",
    "  1999-12-31 11:59:59",
    "\t1999-12-31 11:59:59",
    "\t1999-12-31 11:59:59\n",
    "1999-12-31 11:59:59.",
    "1999-12-31 11:59:59.9",
    " 1999-12-31 11:59:59.9",
    "1999-12-31 11:59:59.99",
    "1999-12-31 11:59:59.999",
    "1999-12-31 11:59:59.9999",
    "1999-12-31 11:59:59.99999",
    "1999-12-31 11:59:59.999999",
    "1999-12-31 11:59:59.9999999",
    "1999-12-31 11:59:59.99999999",
    "1999-12-31 11:59:59.999999999",
    "31/12/1999",
    "31/12/1999 11:59:59.999",
    "1999-12-3",
    "1999-12-31",
    "1999/12/31",
    "1999-12",
    "1999/12",
    "1975/06",
    "1975/06/18",
    "1975/06/18 06:48:57",
    "1999-12-29\n",
    "\t1999-12-30",
    " \n1999-12-31",
    "1999/12/31",
    //TODO: edge cases found during fuzzing that are not fully supported yet
//    "1-1-1",
//    "11-1-1",
//    "111-1-1",
//    "11111-1-1",
    //"1- 1-1"
//    "1- 1-1",
//    "\t3- 4-2",
//    "\t3- 4-2:",
//    "\t3-4- 2:"
  )

  private val dateValues = Seq(
    Date.valueOf("2020-07-24"),
    Date.valueOf("2020-07-25"),
    Date.valueOf("1999-12-31"))

  private val tsValues = Seq(
    Timestamp.valueOf("2015-07-24 10:00:00.3"),
    Timestamp.valueOf("2015-07-25 02:02:02.2"),
    Timestamp.valueOf("1999-12-31 11:59:59.999")
  )
}

