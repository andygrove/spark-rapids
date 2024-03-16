/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

import com.nvidia.spark.rapids.TestUtils.findOperator

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.execution.exchange.ShuffleExchangeLike
import org.apache.spark.sql.functions.broadcast
import org.apache.spark.sql.rapids.execution.{GpuBroadcastHashJoinExec, GpuHashJoin}

class BroadcastHashJoinSuite extends SparkQueryCompareTestSuite {

  test("SPARK-45882: BroadcastHashJoinExec propagate partitioning should respect " +
    "CoalescedHashPartitioning") {

    val conf = new SparkConf()
      .set("spark.sql.optimizer.canChangeCachedPlanOutputPartitioning", "true")

    def doTest(spark: SparkSession): Unit = {

      import spark.implicits._

      val df = (1 to 100)
        .map(i => (i, i.toString))
        .toDF("key", "value")
        .repartition(4)
      df.createOrReplaceTempView("testData")

      val df2 = Seq((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)) // Nil
        .toDF("a", "b")
        .repartition(2)
      df2.createOrReplaceTempView("testData2")

      val cached = spark.sql(
        """
          |select /*+ broadcast(testData) */ key, value, a
          |from testData join (
          | select a from testData2 group by a
          |)tmp on key = a
          |""".stripMargin).cache()

      try {
        // materialize aqe first to propagate output partitioning
        cached.count()
        val df = cached.groupBy("key").count()

        val actual = df.collect().map(_.toString).sorted.mkString("\n")
        assert(actual == "[1,1]\n[2,1]\n[3,1]")

        // assert that there are no shuffles in the plan
        TestUtils.findOperator(df.queryExecution.executedPlan, {
          case _: ShuffleExchangeLike => true
          case _ => false
        }).isEmpty
      } finally {
        cached.unpersist()
      }

    }

    withCpuSparkSession(doTest, conf)
    withGpuSparkSession(doTest, conf)
  }

  test("broadcast hint isn't propagated after a join") {
    val conf = new SparkConf()
      .set("spark.sql.autoBroadcastJoinThreshold", "-1")

    withGpuSparkSession(spark => {
      val df1 = longsDf(spark)
      val df2 = nonZeroLongsDf(spark)

      val df3 = df1.join(broadcast(df2), Seq("longs"), "inner").drop(df2("longs"))
      val df4 = longsDf(spark)
      val df5 = df4.join(df3, Seq("longs"), "inner")

      // execute the plan so that the final adaptive plan is available when AQE is on
      df5.collect()
      val plan = df5.queryExecution.executedPlan

      val bhjCount = PlanUtils.findOperators(plan, _.isInstanceOf[GpuBroadcastHashJoinExec])
      assert(bhjCount.size === 1)

      val shjCount = PlanUtils.findOperators(plan, _.isInstanceOf[GpuShuffledSymmetricHashJoinExec])
      assert(shjCount.size === 1)
    }, conf)
  }

  test("broadcast hint in SQL") {
    withGpuSparkSession(spark => {
      longsDf(spark).createOrReplaceTempView("t")
      longsDf(spark).createOrReplaceTempView("u")

      for (name <- Seq("BROADCAST", "BROADCASTJOIN", "MAPJOIN")) {
        val plan1 = spark.sql(s"SELECT /*+ $name(t) */ * FROM t JOIN u ON t.longs = u.longs")
        val plan2 = spark.sql(s"SELECT /*+ $name(u) */ * FROM t JOIN u ON t.longs = u.longs")

        // execute the plan so that the final adaptive plan is available when AQE is on
        plan1.collect()
        val finalPlan1 = findOperator(plan1.queryExecution.executedPlan,
          _.isInstanceOf[GpuBroadcastHashJoinExec])
        assert(finalPlan1.get.asInstanceOf[GpuHashJoin].buildSide == GpuBuildLeft)

        // execute the plan so that the final adaptive plan is available when AQE is on
        plan2.collect()
        val finalPlan2 = findOperator(plan2.queryExecution.executedPlan,
          _.isInstanceOf[GpuBroadcastHashJoinExec])
        assert(finalPlan2.get.asInstanceOf[GpuHashJoin].buildSide == GpuBuildRight)
      }
    })
  }
}
