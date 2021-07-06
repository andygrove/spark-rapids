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

import ai.rapids.cudf.{ColumnVector, ColumnView, DType, HostColumnVector, HostColumnVectorCore, HostMemoryBuffer}

import java.sql.{Date, Timestamp}
import org.junit.jupiter.api.Assertions.{assertArrayEquals, assertEquals}
import org.scalatest.BeforeAndAfterEach

import scala.collection.mutable.ListBuffer
import scala.util.Random
import org.apache.spark.SparkConf
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.functions.{col, to_date, to_timestamp, unix_timestamp}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.rapids.GpuToTimestamp.{FIX_DATES, FIX_SINGLE_DIGIT_DAY_1, FIX_SINGLE_DIGIT_DAY_2, FIX_SINGLE_DIGIT_MONTH, REMOVE_WHITESPACE_FROM_MONTH_DAY, withResource}
import org.apache.spark.sql.rapids.{GpuToTimestamp, RegexReplace}

import java.text.SimpleDateFormat

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

  test("Regex: Remove whitespace from month and day") {
    testRegex(REMOVE_WHITESPACE_FROM_MONTH_DAY,
    Seq("1- 1-1", "1-1- 1", null),
    Seq("1-1-1", "1-1-1", null))
  }

  test("Regex: Fix single digit month") {
    testRegex(FIX_SINGLE_DIGIT_MONTH,
      Seq("1-2-3", "1111-2-3", null),
      Seq("1-02-3", "1111-02-3", null))
  }

  test("Regex: Fix single digit day 1") {
    // single digit day followed by non digit
    testRegex(FIX_SINGLE_DIGIT_DAY_1,
      Seq("1111-02-3 ", "1111-02-3:", null),
      Seq("1111-02-03 ", "1111-02-03:", null))
  }

  test("Regex: Fix single digit day 2") {
    // single digit day at end of string
    testRegex(FIX_SINGLE_DIGIT_DAY_2,
      Seq("1-02-3", "1111-02-3", "1111-02-03", null),
      Seq("1-02-03", "1111-02-03", "1111-02-03", null))
  }

  test("Regex: Apply all date rules") {
    val testPairs = Seq(
      ("1- 1-1", "1-01-01"),
      ("1-1- 1", "1-01-01"),
      ("1- 1- 1", "1-01-01"),
      ("1999-12-31", "1999-12-31"),
      ("1999-2-31", "1999-02-31"),
      ("1999-2-3", "1999-02-03")
    )
    val values = testPairs.map(_._1)
    val expected = testPairs.map(_._2)
    withResource(ColumnVector.fromStrings(values: _*)) { v =>
      withResource(ColumnVector.fromStrings(expected: _*)) { expected =>
        val actual = FIX_DATES.foldLeft(v.incRefCount())((a, b) => {
          withResource(a) {
            _.stringReplaceWithBackrefs(b.pattern, b.backref)
          }
        })
        withResource(actual) { _ =>
          assertColumnsAreEqual(expected, actual)
        }
      }
    }
  }

//  test("Parse fixed up timestamps") {
//    // check that we can correctly parse strings that have been fixed up to remove whitespace and
//    // to convert single digit month and day to double digit
//    val values = Seq("1-01-01", "1999-12-31")
//    val expected = Seq(12L, 12L)
//    withResource(ColumnVector.fromStrings(values: _*)) { v =>
//      withResource(ColumnVector.fromLongs(expected: _*)) { e =>
//        val actual = GpuToTimestamp.xxx(v, DType.TIMESTAMP_MICROSECONDS, "%Y-%m-%d",
//          (col, strfFormat) => col.asTimestampMicroseconds(strfFormat))
//        assertColumnsAreEqual(e, actual.incRefCount())
//      }
//    }
//  }

  private def testRegex(rule: RegexReplace, values: Seq[String], expected: Seq[String]): Unit = {
    withResource(ColumnVector.fromStrings(values: _*)) { v =>
      withResource(ColumnVector.fromStrings(expected: _*)) { expected =>
        withResource(v.stringReplaceWithBackrefs(rule.pattern, rule.backref)) { actual =>
          assertColumnsAreEqual(expected, actual)
        }
      }
    }
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
    "1- 1-1",
    "1-1- 1",
    "1- 1- 1",
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

  /**
   * Checks and asserts that passed in columns match
   *
   * @param expect The expected result column
   * @param cv     The input column
   */
  def assertColumnsAreEqual(expect: ColumnView, cv: ColumnView): Unit = {
    assertColumnsAreEqual(expect, cv, "unnamed")
  }

  /**
   * Checks and asserts that passed in columns match
   *
   * @param expected The expected result column
   * @param cv       The input column
   * @param colName  The name of the column
   */
  def assertColumnsAreEqual(expected: ColumnView, cv: ColumnView, colName: String): Unit = {
    assertPartialColumnsAreEqual(expected, 0, expected.getRowCount, cv, colName, true)
  }

  /**
   * Checks and asserts that passed in host columns match
   *
   * @param expected The expected result host column
   * @param cv       The input host column
   * @param colName  The name of the host column
   */
  def assertColumnsAreEqual(expected: HostColumnVector,
                            cv: HostColumnVector, colName: String): Unit = {
    assertPartialColumnsAreEqual(expected, 0,
      expected.getRowCount, cv, colName, true)
  }

  // copied from cuDF test suite
  def assertPartialColumnsAreEqual(
                                    expected: ColumnView,
                                    rowOffset: Long,
                                    length: Long,
                                    cv: ColumnView,
                                    colName: String,
                                    enableNullCheck: Boolean): Unit = {
    try {
      val hostExpected = expected.copyToHost
      val hostcv = cv.copyToHost
      try assertPartialColumnsAreEqual(hostExpected, rowOffset, length,
        hostcv, colName, enableNullCheck)
      finally {
        if (hostExpected != null) hostExpected.close()
        if (hostcv != null) hostcv.close()
      }
    }
  }

  // copied from cuDF test suite
  def assertPartialColumnsAreEqual(
                                    expected: HostColumnVectorCore,
                                    rowOffset: Long, length: Long,
                                    cv: HostColumnVectorCore,
                                    colName: String, enableNullCheck: Boolean): Unit = {
    assertEquals(expected.getType, cv.getType, "Type For Column " + colName)
    assertEquals(length, cv.getRowCount, "Row Count For Column " + colName)
    assertEquals(expected.getNumChildren, cv.getNumChildren, "Child Count for Column " + colName)
    if (enableNullCheck) assertEquals(expected.getNullCount,
      cv.getNullCount, "Null Count For Column " + colName)
    else {
      // TODO add in a proper check when null counts are
      //  supported by serializing a partitioned column
    }

    import ai.rapids.cudf.DType.DTypeEnum._

    val `type`: DType = expected.getType
    for (expectedRow <- rowOffset until (rowOffset + length)) {
      val tableRow: Long = expectedRow - rowOffset
      assertEquals(expected.isNull(expectedRow), cv.isNull(tableRow),
        "NULL for Column " + colName + " Row " + tableRow)
      if (!expected.isNull(expectedRow)) `type`.getTypeId match {
        case BOOL8 => // fall through

        case INT8 =>
        case UINT8 =>
          assertEquals(expected.getByte(expectedRow), cv.getByte(tableRow),
            "Column " + colName + " Row " + tableRow)

        case INT16 =>
        case UINT16 =>
          assertEquals(expected.getShort(expectedRow), cv.getShort(tableRow),
            "Column " + colName + " Row " + tableRow)

        case INT32 =>
        case UINT32 =>
        case TIMESTAMP_DAYS =>
        case DURATION_DAYS =>
        case DECIMAL32 =>
          assertEquals(expected.getInt(expectedRow), cv.getInt(tableRow),
            "Column " + colName + " Row " + tableRow)

        case INT64 =>
        case UINT64 =>
        case DURATION_MICROSECONDS =>
        case DURATION_MILLISECONDS =>
        case DURATION_NANOSECONDS =>
        case DURATION_SECONDS =>
        case TIMESTAMP_MICROSECONDS =>
        case TIMESTAMP_MILLISECONDS =>
        case TIMESTAMP_NANOSECONDS =>
        case TIMESTAMP_SECONDS =>
        case DECIMAL64 =>
          assertEquals(expected.getLong(expectedRow), cv.getLong(tableRow),
            "Column " + colName + " Row " + tableRow)

        //        case FLOAT32 =>
        //          assertEqualsWithinPercentage(expected.getFloat(expectedRow),
        //                  cv.getFloat(tableRow), 0.0001,
        //                  "Column " + colName + " Row " + tableRow)
        //
        //        case FLOAT64 =>
        //          assertEqualsWithinPercentage(expected.getDouble(expectedRow),
        //                  cv.getDouble(tableRow), 0.0001,
        //                  "Column " + colName + " Row " + tableRow)

        case STRING =>
          assertArrayEquals(expected.getUTF8(expectedRow), cv.getUTF8(tableRow),
            "Column " + colName + " Row " + tableRow)

        case LIST =>
          val expectedOffsets: HostMemoryBuffer = expected.getOffsets
          val cvOffsets: HostMemoryBuffer = cv.getOffsets
          val expectedChildRows: Int = expectedOffsets.getInt((expectedRow + 1) * 4) -
            expectedOffsets.getInt(expectedRow * 4)
          val cvChildRows: Int = cvOffsets.getInt((tableRow + 1) * 4) -
            cvOffsets.getInt(tableRow * 4)
          assertEquals(expectedChildRows, cvChildRows,
            "Child row count for Column " + colName + " Row " + tableRow)

        case STRUCT =>
        // parent column only has validity which was checked above

        case _ =>
          throw new IllegalArgumentException(`type` + " is not supported yet")
      }
    }
  }

}

