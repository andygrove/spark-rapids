/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

import org.apache.spark.sql.catalyst.expressions.{Ascending, Attribute, AttributeReference, Expression, InputFileBlockLength, InputFileBlockStart, InputFileName, SortOrder}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution._
import org.apache.spark.sql.execution.adaptive.{AdaptiveSparkPlanExec, BroadcastQueryStageExec, CustomShuffleReaderExec, QueryStageExec, ShuffleQueryStageExec}
import org.apache.spark.sql.execution.columnar.InMemoryTableScanExec
import org.apache.spark.sql.execution.command.ExecutedCommandExec
import org.apache.spark.sql.execution.datasources.v2.DataSourceV2ScanExecBase
import org.apache.spark.sql.execution.exchange.{BroadcastExchangeLike, Exchange, ReusedExchangeExec}
import org.apache.spark.sql.execution.joins.{BroadcastHashJoinExec, BroadcastNestedLoopJoinExec}
import org.apache.spark.sql.rapids.{GpuDataSourceScanExec, GpuFileSourceScanExec, GpuInputFileBlockLength, GpuInputFileBlockStart, GpuInputFileName, GpuShuffleEnv}
import org.apache.spark.sql.rapids.execution.{GpuBroadcastExchangeExecBase, GpuBroadcastToCpuExec, GpuCustomShuffleReaderExec, GpuHashJoin, GpuShuffleExchangeExecBase}

/**
 * Rules that run after the row to columnar and columnar to row transitions have been inserted.
 * These rules insert transitions to and from the GPU, and then optimize various transitions.
 */
class GpuTransitionOverrides extends Rule[SparkPlan] {
  var conf: RapidsConf = null

  def optimizeGpuPlanTransitions(plan: SparkPlan): SparkPlan = plan match {
    case HostColumnarToGpu(r2c: RowToColumnarExec, goal) =>
      GpuRowToColumnarExec(optimizeGpuPlanTransitions(r2c.child), goal)
    case ColumnarToRowExec(bb: GpuBringBackToHost) =>
      getColumnarToRowExec(optimizeGpuPlanTransitions(bb.child))
    case p =>
      p.withNewChildren(p.children.map(optimizeGpuPlanTransitions))
  }

  private def getColumnarToRowExec(plan: SparkPlan, exportColumnRdd: Boolean = false) = {
    ShimLoader.getSparkShims.getGpuColumnarToRowTransition(plan, exportColumnRdd)
  }

  /** Adds the appropriate coalesce after a shuffle depending on the type of shuffle configured */
  private def addPostShuffleCoalesce(plan: SparkPlan): SparkPlan = {
    if (GpuShuffleEnv.isRapidsShuffleEnabled) {
      GpuCoalesceBatches(plan, TargetSize(conf.gpuTargetBatchSizeBytes))
    } else {
      GpuShuffleCoalesceExec(plan, conf.gpuTargetBatchSizeBytes)
    }
  }

  def optimizeAdaptiveTransitions(
      plan: SparkPlan,
      parent: Option[SparkPlan]): SparkPlan = plan match {
    // HostColumnarToGpu(RowToColumnarExec(..)) => GpuRowToColumnarExec(..)
    case HostColumnarToGpu(r2c: RowToColumnarExec, goal) =>
      val transition = GpuRowToColumnarExec(
        optimizeAdaptiveTransitions(r2c.child, Some(r2c)), goal)
      r2c.child match {
        case _: AdaptiveSparkPlanExec =>
          // When the input is an adaptive plan we do not get to see the GPU version until
          // the plan is executed and sometimes the plan will have a GpuColumnarToRowExec as the
          // final operator and we can bypass this to keep the data columnar by inserting
          // the [[AvoidAdaptiveTransitionToRow]] operator here
          ShimLoader.getSparkShims.createAvoidAdaptiveTransitionToRow(transition)
        case _ =>
          transition
      }

    case ColumnarToRowExec(GpuBringBackToHost(
        GpuShuffleCoalesceExec(e: GpuShuffleExchangeExecBase, _))) if parent.isEmpty =>
      // We typically want the final operator in the plan (the operator that has no parent) to be
      // wrapped in `ColumnarToRowExec(GpuBringBackToHost(ShuffleCoalesceExec(_)))` operators to
      // bring the data back onto the host and be translated to rows so that it can be returned
      // from the Spark API. However, in the case of AQE, each exchange operator is treated as an
      // individual query with no parent and we need to remove these operators in this case
      // because we need to return an operator that implements `BroadcastExchangeLike` or
      // `ShuffleExchangeLike`. The coalesce step gets added back into the plan later on, in a
      // future query stage that reads the output from this query stage. This is handled in the
      // case clauses below.
      e.withNewChildren(e.children.map(c => optimizeAdaptiveTransitions(c, Some(e))))

    case ColumnarToRowExec(GpuBringBackToHost(
        GpuCoalesceBatches(e: GpuShuffleExchangeExecBase, _))) if parent.isEmpty =>
      // We typically want the final operator in the plan (the operator that has no parent) to be
      // wrapped in `ColumnarToRowExec(GpuBringBackToHost(GpuCoalesceBatches(_)))` operators to
      // bring the data back onto the host and be translated to rows so that it can be returned
      // from the Spark API. However, in the case of AQE, each exchange operator is treated as an
      // individual query with no parent and we need to remove these operators in this case
      // because we need to return an operator that implements `BroadcastExchangeLike` or
      // `ShuffleExchangeLike`. The coalesce step gets added back into the plan later on, in a
      // future query stage that reads the output from this query stage. This is handled in the
      // case clauses below.
      e.withNewChildren(e.children.map(c => optimizeAdaptiveTransitions(c, Some(e))))

    case s: ShuffleQueryStageExec =>
      // When reading a materialized shuffle query stage in AQE mode, we need to insert an
      // operator to coalesce batches. We either insert it directly around the shuffle query
      // stage, or around the custom shuffle reader, if one exists.
      val plan = GpuTransitionOverrides.getNonQueryStagePlan(s)
      if (plan.supportsColumnar && plan.isInstanceOf[GpuExec]) {
        parent match {
          case Some(_: GpuCustomShuffleReaderExec) =>
            // We can't insert a coalesce batches operator between a custom shuffle reader
            // and a shuffle query stage, so we instead insert it around the custom shuffle
            // reader later on, in the next top-level case clause.
            s
          case _ =>
            // Directly wrap shuffle query stage with coalesce batches operator
            addPostShuffleCoalesce(s)
        }
      } else {
        s
      }

    case e: GpuCustomShuffleReaderExec =>
      // We wrap custom shuffle readers with a coalesce batches operator here.
      addPostShuffleCoalesce(e.copy(child = optimizeAdaptiveTransitions(e.child, Some(e))))

    // Query stages that have already executed on the GPU could be used by CPU operators
    // in future query stages. Note that because these query stages have already executed, we
    // don't need to recurse down and optimize them again
    case ColumnarToRowExec(e: BroadcastQueryStageExec) =>
      e.plan match {
        case ReusedExchangeExec(_, b: GpuBroadcastExchangeExecBase) =>
          // we can't directly re-use a GPU broadcast exchange to feed a CPU broadcast
          // hash join but Spark will sometimes try and do this (see
          // https://issues.apache.org/jira/browse/SPARK-35093 for more information) so we
          // need to convert the output to rows in the driver before broadcasting the data
          // to the executors
          GpuBroadcastToCpuExec(b.mode, b.child)
        case _ => getColumnarToRowExec(e)
      }
    case ColumnarToRowExec(e: ShuffleQueryStageExec) =>
      getColumnarToRowExec(optimizeAdaptiveTransitions(e, Some(plan)))

    case ColumnarToRowExec(bb: GpuBringBackToHost) =>
      optimizeAdaptiveTransitions(bb.child, Some(bb)) match {
        case e: GpuBroadcastExchangeExecBase => e
        case e: GpuShuffleExchangeExecBase => e
        case other => getColumnarToRowExec(other)
      }

    case p: AdaptiveSparkPlanExec =>
      GpuColumnarToRowExec(p, false)

    case p =>
      p.withNewChildren(p.children.map(c => optimizeAdaptiveTransitions(c, Some(p))))
  }

  private def isGpuShuffleLike(execNode: SparkPlan): Boolean =
    execNode.isInstanceOf[GpuShuffleExchangeExecBase] ||
      execNode.isInstanceOf[GpuCustomShuffleReaderExec]

  /**
   * This optimizes the plan to remove [[GpuCoalesceBatches]] nodes that are unnecessary
   * or undesired in some situations.
   *
   * @note This does not examine [[GpuShuffleCoalesceExec]] nodes in the plan, as they
   *       are always required after GPU columnar exchanges during normal shuffle
   *       to place the data after shuffle on the GPU. Those nodes also do not
   *       coalesce to the same goal as used by [[GpuCoalesceBatches]], so a
   *       [[GpuShuffleCoalesceExec]] immediately followed by a [[GpuCoalesceBatches]] is
   *       not unusual.
   */
  def optimizeCoalesce(plan: SparkPlan): SparkPlan = plan match {
    case c2r @ GpuColumnarToRowExecParent(gpuCoalesce: GpuCoalesceBatches, _)
      if !isGpuShuffleLike(gpuCoalesce.child) =>
        // Don't build a batch if we are just going to go back to ROWS
        // and there isn't a GPU shuffle involved
        c2r.withNewChildren(gpuCoalesce.children.map(optimizeCoalesce))
    case GpuCoalesceBatches(r2c: GpuRowToColumnarExec, goal: TargetSize) =>
      // TODO in the future we should support this for all goals, but
      // GpuRowToColumnarExec preallocates all of the memory, and the builder does not
      // support growing the sizes dynamically....

      // Don't build batches and then coalesce, just build the right sized batch
      GpuRowToColumnarExec(optimizeCoalesce(r2c.child),
        CoalesceGoal.maxRequirement(goal, r2c.goal).asInstanceOf[CoalesceSizeGoal])
    case GpuCoalesceBatches(co: GpuCoalesceBatches, goal) =>
      GpuCoalesceBatches(optimizeCoalesce(co.child), CoalesceGoal.maxRequirement(goal, co.goal))
    case GpuCoalesceBatches(child: GpuExec, goal)
      if CoalesceGoal.satisfies(child.outputBatching, goal) =>
      // The goal is already satisfied so remove the batching
      child.withNewChildren(child.children.map(optimizeCoalesce))
    case p =>
      p.withNewChildren(p.children.map(optimizeCoalesce))
  }

  private def insertCoalesce(plans: Seq[SparkPlan], goals: Seq[CoalesceGoal],
      disableUntilInput: Boolean): Seq[SparkPlan] = {
    plans.zip(goals).map {
      case (plan, null) =>
        // No coalesce requested
        insertCoalesce(plan, disableUntilInput)
      case (plan, goal @ RequireSingleBatch) =>
        // Even if coalesce is disabled a single batch is required to make this operator work
        // This should not cause bugs because we require a single batch in situations where
        // Spark also buffers data, so any operator that needs coalesce disabled would also
        // get an incorrect answer in regular Spark
        GpuCoalesceBatches(insertCoalesce(plan, disableUntilInput), goal)
      case (plan, _) if disableUntilInput =>
        // We wanted to coalesce the input but cannot because it could cause errors
        insertCoalesce(plan, disableUntilInput)
      case (plan, goal) =>
        GpuCoalesceBatches(insertCoalesce(plan, disableUntilInput), goal)
    }
  }

  /**
   * Essentially check if this plan is in the same task as a file input.
   */
  private def hasDirectLineToInput(plan: SparkPlan): Boolean = plan match {
    case _: Exchange => false
    case _: DataSourceScanExec => true
    case _: GpuDataSourceScanExec => true
    case _: DataSourceV2ScanExecBase => true
    case _: RDDScanExec => true // just in case an RDD was reading in data
    case p => p.children.exists(hasDirectLineToInput)
  }

  /**
   * Essentially check if we have hit a boundary of a task.
   */
  private def shouldEnableCoalesce(plan: SparkPlan): Boolean = plan match {
    case _: Exchange => true
    case _: DataSourceScanExec => true
    case _: GpuDataSourceScanExec => true
    case _: DataSourceV2ScanExecBase => true
    case _: RDDScanExec => true // just in case an RDD was reading in data
    case _ => false
  }

  /**
   * Because we cannot change the executors in spark itself we need to try and account for
   * the ones that might have issues with coalesce here.
   */
  private def disableCoalesceUntilInput(exec: Expression): Boolean = exec match {
    case _: InputFileName => true
    case _: InputFileBlockStart => true
    case _: InputFileBlockLength => true
    case e => e.children.exists(disableCoalesceUntilInput)
  }

  /**
   * Because we cannot change the executors in spark itself we need to try and account for
   * the ones that might have issues with coalesce here.
   */
  private def disableCoalesceUntilInput(plan: SparkPlan): Boolean = {
    plan.expressions.exists(disableCoalesceUntilInput)
  }

  private def disableScanUntilInput(exec: Expression): Boolean = {
    exec match {
      case _: InputFileName => true
      case _: InputFileBlockStart => true
      case _: InputFileBlockLength => true
      case _: GpuInputFileName => true
      case _: GpuInputFileBlockStart => true
      case _: GpuInputFileBlockLength => true
      case e => e.children.exists(disableScanUntilInput)
    }
  }

  private def disableScanUntilInput(plan: SparkPlan): Boolean = {
    plan.expressions.exists(disableScanUntilInput)
  }

  // This walks from the output to the input to look for any uses of InputFileName,
  // InputFileBlockStart, or InputFileBlockLength when we use a Parquet read because
  // we can't support the coalesce file reader optimization when this is used.
  private def updateScansForInput(plan: SparkPlan,
      disableUntilInput: Boolean = false): SparkPlan = plan match {
    case batchScan: GpuBatchScanExec =>
      if (batchScan.scan.isInstanceOf[GpuParquetScanBase] &&
        (disableUntilInput || disableScanUntilInput(batchScan))) {
        ShimLoader.getSparkShims.copyParquetBatchScanExec(batchScan, true)
      } else {
        batchScan
      }
    case fileSourceScan: GpuFileSourceScanExec =>
      if ((disableUntilInput || disableScanUntilInput(fileSourceScan))) {
        ShimLoader.getSparkShims.copyFileSourceScanExec(fileSourceScan, true)
      } else {
        fileSourceScan
      }
    case p =>
      val planDisableUntilInput = disableScanUntilInput(p) && hasDirectLineToInput(p)
      p.withNewChildren(p.children.map(c => {
        updateScansForInput(c, planDisableUntilInput || disableUntilInput)
      }))
  }

  // This walks from the output to the input so disableUntilInput can walk its way from when
  // we hit something that cannot allow for coalesce up until the input
  private def insertCoalesce(plan: SparkPlan,
      disableUntilInput: Boolean = false): SparkPlan = plan match {
    case exec: GpuExec =>
      // We will disable coalesce if it is already disabled and we cannot re-enable it
      val shouldDisable = (disableUntilInput && !shouldEnableCoalesce(exec)) ||
        //or if we should disable it and it is in a stage with a file input that would matter
        (exec.disableCoalesceUntilInput() && hasDirectLineToInput(exec))
      val tmp = exec.withNewChildren(insertCoalesce(exec.children, exec.childrenCoalesceGoal,
        shouldDisable))
      if (exec.coalesceAfter && !shouldDisable) {
        GpuCoalesceBatches(tmp, TargetSize(conf.gpuTargetBatchSizeBytes))
      } else {
        tmp
      }
    case p =>
      // We will disable coalesce if it is already disabled and we cannot re-enable it
      val shouldDisable = disableUntilInput && !shouldEnableCoalesce(p)  ||
        //or if we should disable it and it is in a stage with a file input that would matter
        (disableCoalesceUntilInput(p) && hasDirectLineToInput(p))
      p.withNewChildren(p.children.map(c => insertCoalesce(c, shouldDisable)))
  }

  /**
   * Inserts a shuffle coalesce after every shuffle to coalesce the serialized tables
   * on the host before copying the data to the GPU.
   * @note This should not be used in combination with the RAPIDS shuffle.
   */
  private def insertShuffleCoalesce(plan: SparkPlan): SparkPlan = plan match {
    case exec: GpuShuffleExchangeExecBase =>
      // always follow a GPU shuffle with a shuffle coalesce
      GpuShuffleCoalesceExec(exec.withNewChildren(exec.children.map(insertShuffleCoalesce)),
        conf.gpuTargetBatchSizeBytes)
    case exec => exec.withNewChildren(plan.children.map(insertShuffleCoalesce))
  }

  /**
   * Inserts a transition to be running on the CPU columnar
   */
  private def insertColumnarFromGpu(plan: SparkPlan): SparkPlan = {
    if (plan.supportsColumnar && plan.isInstanceOf[GpuExec]) {
      GpuBringBackToHost(insertColumnarToGpu(plan))
    } else {
      plan.withNewChildren(plan.children.map(insertColumnarFromGpu))
    }
  }

  /**
   * Inserts a transition to be running on the GPU from CPU columnar
   */
  private def insertColumnarToGpu(plan: SparkPlan): SparkPlan = {
    val nonQueryStagePlan = GpuTransitionOverrides.getNonQueryStagePlan(plan)
    if (nonQueryStagePlan.supportsColumnar && !nonQueryStagePlan.isInstanceOf[GpuExec]) {
      HostColumnarToGpu(insertColumnarFromGpu(plan), TargetSize(conf.gpuTargetBatchSizeBytes))
    } else {
      plan.withNewChildren(plan.children.map(insertColumnarToGpu))
    }
  }

  private def insertHashOptimizeSorts(plan: SparkPlan): SparkPlan = {
    if (conf.enableHashOptimizeSort) {
      // Insert a sort after the last hash-based op before the query result if there are no
      // intermediate nodes that have a specified sort order. This helps with the size of
      // Parquet and Orc files
      plan match {
        case _: GpuHashJoin =>
          val sortOrder = getOptimizedSortOrder(plan)
          GpuSortExec(sortOrder, false, plan, SortEachBatch)
        case _: GpuHashAggregateExec =>
          val sortOrder = getOptimizedSortOrder(plan)
          GpuSortExec(sortOrder, false, plan, SortEachBatch)
        case p =>
          if (p.outputOrdering.isEmpty) {
            plan.withNewChildren(plan.children.map(insertHashOptimizeSorts))
          } else {
            plan
          }
      }
    } else {
      plan
    }
  }

  private def getOptimizedSortOrder(plan: SparkPlan): Seq[SortOrder] = {
    plan.output.map { expr =>
      val wrapped = GpuOverrides.wrapExpr(expr, conf, None)
      wrapped.tagForGpu()
      assert(wrapped.canThisBeReplaced)
      ShimLoader.getSparkShims.sortOrder(
        wrapped.convertToGpu(),
        Ascending,
        Ascending.defaultNullOrdering)
    }
  }

  def assertIsOnTheGpu(exp: Expression, conf: RapidsConf): Unit = {
    // There are no GpuAttributeReference or GpuSortOrder
    if (!exp.isInstanceOf[AttributeReference] &&
        !exp.isInstanceOf[SortOrder] &&
        !exp.isInstanceOf[GpuExpression] &&
      !conf.testingAllowedNonGpu.contains(PlanUtils.getBaseNameFromClass(exp.getClass.toString))) {
      throw new IllegalArgumentException(s"The expression $exp is not columnar ${exp.getClass}")
    }
    exp.children.foreach(subExp => assertIsOnTheGpu(subExp, conf))
  }

  def assertIsOnTheGpu(plan: SparkPlan, conf: RapidsConf): Unit = {
    val isAdaptiveEnabled = plan.conf.adaptiveExecutionEnabled
    plan match {
      case _: BroadcastExchangeLike if isAdaptiveEnabled =>
        // broadcasts are left on CPU for now when AQE is enabled
      case _: BroadcastHashJoinExec | _: BroadcastNestedLoopJoinExec
          if isAdaptiveEnabled =>
        // broadcasts are left on CPU for now when AQE is enabled
      case _: AdaptiveSparkPlanExec | _: QueryStageExec | _: CustomShuffleReaderExec =>
        // we do not yet fully support GPU-acceleration when AQE is enabled, so we skip checking
        // the plan in this case - https://github.com/NVIDIA/spark-rapids/issues/5
      case lts: LocalTableScanExec =>
        if (!lts.expressions.forall(_.isInstanceOf[AttributeReference])) {
          throw new IllegalArgumentException("It looks like some operations were " +
            s"pushed down to LocalTableScanExec ${lts.expressions.mkString(",")}")
        }
      case imts: InMemoryTableScanExec =>
        if (!imts.expressions.forall(_.isInstanceOf[AttributeReference])) {
          throw new IllegalArgumentException("It looks like some operations were " +
            s"pushed down to InMemoryTableScanExec ${imts.expressions.mkString(",")}")
        }
      case _: ExecutedCommandExec => () // Ignored
      case _: RDDScanExec => () // Ignored
      case _ =>
        if (!plan.supportsColumnar &&
            // There are some python execs that are not columnar because of a little
            // used feature. This prevents those from failing tests. This also allows
            // the columnar to row transitions to not cause test issues because they too
            // are not columnar (they output rows) but are instances of GpuExec.
            !plan.isInstanceOf[GpuExec] && 
            !conf.testingAllowedNonGpu.exists(nonGpuClass => 
                PlanUtils.sameClass(plan, nonGpuClass))) {
          throw new IllegalArgumentException(s"Part of the plan is not columnar " +
            s"${plan.getClass}\n${plan}")
        }
        // filter out the output expressions since those are not GPU expressions
        val planOutput = plan.output.toSet
        // avoid checking expressions of GpuFileSourceScanExec since all expressions are
        // processed by driver and not run on GPU.
        if (!plan.isInstanceOf[GpuFileSourceScanExec]) {
          plan.expressions.filter(_ match {
            case a: Attribute => !planOutput.contains(a)
            case _ => true
          }).foreach(assertIsOnTheGpu(_, conf))
        }
    }
    plan.children.foreach(assertIsOnTheGpu(_, conf))
  }

  /**
   * This is intended for testing only and this only supports looking for an exec once.
   */
  private def validateExecsInGpuPlan(plan: SparkPlan, conf: RapidsConf): Unit = {
    val validateExecs = conf.validateExecsInGpuPlan.toSet
    if (validateExecs.nonEmpty) {
      def planContainsInstanceOf(plan: SparkPlan): Boolean = {
        validateExecs.contains(plan.getClass.getSimpleName)
      }
      // to set to make uniq execs
      val execsFound = PlanUtils.findOperators(plan, planContainsInstanceOf).toSet
      val execsNotFound = validateExecs.diff(execsFound.map(_.getClass().getSimpleName))
      require(execsNotFound.isEmpty,
        s"Plan ${plan.toString()} does not contain the following execs: " +
        execsNotFound.mkString(","))
    }
  }

  def detectAndTagFinalColumnarOutput(plan: SparkPlan): SparkPlan = plan match {
    case d: DeserializeToObjectExec if d.child.isInstanceOf[GpuColumnarToRowExecParent] =>
      val gpuColumnar = d.child.asInstanceOf[GpuColumnarToRowExecParent]
      plan.withNewChildren(Seq(getColumnarToRowExec(gpuColumnar.child, true)))
    case _ => plan
  }

  override def apply(plan: SparkPlan): SparkPlan = {
    this.conf = new RapidsConf(plan.conf)
    if (conf.isSqlEnabled) {
      var updatedPlan = insertHashOptimizeSorts(plan)
      updatedPlan = updateScansForInput(updatedPlan)

      if (Thread.currentThread().getStackTrace.exists(_.getClassName.contains("InsertAdaptive"))) {
        println("Planning an adaptive query")
      } else {
        updatedPlan = insertColumnarFromGpu(updatedPlan)
      }

      updatedPlan = insertCoalesce(updatedPlan)
      // only insert shuffle coalesces when using normal shuffle
      if (!GpuShuffleEnv.isRapidsShuffleEnabled) {
        updatedPlan = insertShuffleCoalesce(updatedPlan)
      }
      if (plan.conf.adaptiveExecutionEnabled) {
        updatedPlan = optimizeAdaptiveTransitions(updatedPlan, None)
      } else {
        updatedPlan = optimizeGpuPlanTransitions(updatedPlan)
      }
      updatedPlan = optimizeCoalesce(updatedPlan)
      if (conf.exportColumnarRdd) {
        updatedPlan = detectAndTagFinalColumnarOutput(updatedPlan)
      }
      if (conf.isTestEnabled) {
        assertIsOnTheGpu(updatedPlan, conf)
        // Generate the canonicalized plan to ensure no incompatibilities.
        // The plan itself is not currently checked.
        updatedPlan.canonicalized
        validateExecsInGpuPlan(updatedPlan, conf)
      }
      updatedPlan
    } else {
      plan
    }
  }
}

object GpuTransitionOverrides {
  /**
   * Returning the underlying plan of a query stage, or the plan itself if it is not a
   * query stage. This method is typically used when we want to determine if a plan is
   * a GpuExec or not, and this gets hidden by the query stage wrapper.
   */
  def getNonQueryStagePlan(plan: SparkPlan): SparkPlan = {
    plan match {
      case BroadcastQueryStageExec(_, ReusedExchangeExec(_, plan)) => plan
      case BroadcastQueryStageExec(_, plan) => plan
      case ShuffleQueryStageExec(_, ReusedExchangeExec(_, plan)) => plan
      case ShuffleQueryStageExec(_, plan) => plan
      case _ => plan
    }
  }
}


