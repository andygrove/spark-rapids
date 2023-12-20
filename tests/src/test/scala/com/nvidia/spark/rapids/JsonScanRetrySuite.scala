/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

import ai.rapids.cudf.{ColumnVector, DType, JSONOptions, Schema, Table}
import com.nvidia.spark.rapids.jni.RmmSpark

import org.apache.spark.sql.catalyst.json.rapids.JsonPartitionReader
import org.apache.spark.sql.types._

import scala.util.Random

trait Gen {
  def gen(): String
}

case class DateGen(r: Random) extends Gen {
  override def gen(): String = {
    s"${1000 + r.nextInt(2000)}-${r.nextInt(12)}-${r.nextInt(31)}"
  }
}

case class IntGen(r: Random) extends Gen {
  override def gen(): String = r.nextInt().toString
}

class JsonScanRetrySuite extends RmmSparkRetrySuiteBase {

  test("from_json OOM int") {
    doTest(100000000, IntGen(new Random(42)), DataTypes.IntegerType)
    // fail:
    //doTest(70000000, IntGen(new Random(42)), DataTypes.IntegerType)
  }

  //
//  test("from_json OOM bool") {
//    doTest(10000000, "true", DataTypes.BooleanType)
//  }

  test("from_json OOM date") {
    doTest(25000000, DateGen(new Random(42)), DataTypes.DateType)
    // failure:
    //doTest(30000000, DateGen(new Random(42)), DataTypes.DateType)
  }

  def doTest(n: Int, gen: Gen, toType: DataType): Unit = {

    val b = Schema.builder()
    b.column(DType.STRING, "a");
    val schema = b.build()

    val t1 = System.currentTimeMillis()

    val json = new StringBuilder(n)
    var counter = 0
    while (json.length < n - 1000) {
      json.append("{ \"a\": " + gen.gen() + " }\n")
      counter += 1
    }
    val bytes = json.toString.getBytes

    val t2 = System.currentTimeMillis()

    println(s"Generated ${bytes.length} bytes of input JSON with $counter rows in ${t2-t1} millis")

    val opts = JSONOptions.builder().withLines(true).build()

    val table = Table.readJSON(schema, opts, bytes)
    val t3 = System.currentTimeMillis()

    println("Loaded table with " + table.getDeviceMemorySize + s" bytes in ${t3-t2} millis")

    val col: ColumnVector = table.getColumn(0)
    println("Cast input mem size = " + col.getDeviceMemorySize + " bytes")

    val col2 = GpuCast.doCast(col, DataTypes.StringType, toType,
      CastOptions.DEFAULT_CAST_OPTIONS)
    val t4 = System.currentTimeMillis()

    println("Cast output mem size = " +
      col2.getDeviceMemorySize + s" bytes in ${t4-t3} millis")

    println("Total mem size = " +
      (col.getDeviceMemorySize + col2.getDeviceMemorySize) + " bytes")

    col2.close()
    table.close()
  }

  test("test simple retry") {
    val bufferer = HostLineBuffererFactory.createBufferer(100, Array('\n'.toByte))
    bufferer.add("{\"a\": 1, \"b\": 2".getBytes, 0, 14)

    val cudfSchema = GpuColumnVector.from(StructType(Seq(StructField("a", IntegerType),
      StructField("b", IntegerType))))
    val opts = JSONOptions.builder().withLines(true).build()
    RmmSpark.forceRetryOOM(RmmSpark.getCurrentThreadId)
    val table = JsonPartitionReader.readToTable(bufferer, cudfSchema, NoopMetric,
      opts, "JSON", null)
    table.close()
    // We don't have any good way to verify that the retry was thrown, but we are going to trust
    // that it was.
  }
}
