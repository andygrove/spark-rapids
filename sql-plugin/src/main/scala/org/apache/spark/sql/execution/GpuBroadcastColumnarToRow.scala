package org.apache.spark.sql.execution

import java.util.concurrent.{Callable, Future}
import java.util.UUID

import scala.concurrent.Promise
import scala.util.control.NonFatal

import ai.rapids.cudf.NvtxColor
import com.nvidia.spark.rapids.{GpuExec, GpuMetric, MetricRange, NvtxWithMetrics}
import org.apache.spark.SparkException

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.launcher.SparkLauncher
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.{Attribute, SortOrder}
import org.apache.spark.sql.catalyst.plans.physical.Partitioning
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.execution.exchange.ReusedExchangeExec
import org.apache.spark.sql.execution.metric.SQLMetrics
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.rapids.execution.{GpuBroadcastExchangeExec, GpuBroadcastExchangeExecBase, SerializeBatchDeserializeHostBuffer, SerializeConcatHostBuffersDeserializeBatch}

case class GpuBroadcastColumnarToRow(
      child: SparkPlan,
      exportColumnarRdd: Boolean)
    extends UnaryExecNode with GpuExec {

  import GpuMetric._
  // We need to do this so the assertions don't fail
  override def supportsColumnar = false

  override def output: Seq[Attribute] = child.output

  override def outputPartitioning: Partitioning = child.outputPartitioning

  override def outputOrdering: Seq[SortOrder] = child.outputOrdering

  // Override the original metrics to remove NUM_OUTPUT_BATCHES, which makes no sense.
  override lazy val allMetrics: Map[String, GpuMetric] = Map(
    NUM_OUTPUT_ROWS -> createMetric(outputRowsLevel, DESCRIPTION_NUM_OUTPUT_ROWS),
    TOTAL_TIME -> createNanoTimingMetric(MODERATE_LEVEL, DESCRIPTION_TOTAL_TIME),
    NUM_INPUT_BATCHES -> createMetric(DEBUG_LEVEL, DESCRIPTION_NUM_INPUT_BATCHES))

  @transient
  private lazy val promise = Promise[Broadcast[Any]]()

  /**
   * For registering callbacks on `relationFuture`.
   * Note that calling this field will not start the execution of broadcast job.
   */
  @transient
  lazy val completionFuture: concurrent.Future[Broadcast[Any]] = promise.future

  val _runId: UUID = UUID.randomUUID()

  @transient
  lazy val relationFuture: Future[Broadcast[Any]] = {
    // relationFuture is used in "doExecute". Therefore we can get the execution id correctly here.
    val executionId = sparkContext.getLocalProperty(SQLExecution.EXECUTION_ID_KEY)
    val numOutputBatches = gpuLongMetric(NUM_OUTPUT_BATCHES)
    val numOutputRows = gpuLongMetric(NUM_OUTPUT_ROWS)
    val totalTime = gpuLongMetric(TOTAL_TIME)
    val collectTime = gpuLongMetric(COLLECT_TIME)
    val buildTime = gpuLongMetric(BUILD_TIME)
    val broadcastTime = gpuLongMetric("broadcastTime")

    val task = new Callable[Broadcast[Any]]() {
      override def call(): Broadcast[Any] = {
        // This will run in another thread. Set the execution id so that we can connect these jobs
        // with the correct execution.
        SQLExecution.withExecutionId(sqlContext.sparkSession, executionId) {
          val totalRange = new MetricRange(totalTime)
          try {
            // Setup a job group here so later it may get cancelled by groupId if necessary.
            sparkContext.setJobGroup(_runId.toString, s"broadcast exchange (runId ${_runId})",
              interruptOnCancel = true)
            val collectRange = new NvtxWithMetrics("broadcast collect", NvtxColor.GREEN,
              collectTime)
            val batch = try {
              val data = child.executeColumnar().map(cb => try {
                new SerializeBatchDeserializeHostBuffer(cb)
              } finally {
                cb.close()
              })
              val d = data.collect()
              new SerializeConcatHostBuffersDeserializeBatch(d, output)
            } finally {
              collectRange.close()
            }

            val numRows = batch.numRows
            if (numRows >= 512000000) {
              throw new SparkException(
                s"Cannot broadcast the table with 512 million or more rows: $numRows rows")
            }
            numOutputBatches += 1
            numOutputRows += numRows

            val buildRange = new NvtxWithMetrics("broadcast build", NvtxColor.DARK_GREEN, buildTime)
            try {
              // we only support hashjoin so this is a noop
              // val relation = mode.transform(input, Some(numRows))
              val dataSize = batch.dataSize

              gpuLongMetric("dataSize") += dataSize
              if (dataSize >= (8L << 30)) {
                throw new SparkException(
                  s"Cannot broadcast the table that is larger than 8GB: ${dataSize >> 30} GB")
              }
            } finally {
              buildRange.close()
            }
            val broadcastRange = new NvtxWithMetrics("broadcast", NvtxColor.CYAN, broadcastTime)
            val broadcasted = try {
              // Broadcast the relation
              sparkContext.broadcast(batch.asInstanceOf[Any])
            } finally {
              broadcastRange.close()
            }

            SQLMetrics.postDriverMetricUpdates(sparkContext, executionId, metrics.values.toSeq)
            promise.success(broadcasted)
            broadcasted
          } catch {
            // SPARK-24294: To bypass scala bug: https://github.com/scala/bug/issues/9554, we throw
            // SparkFatalException, which is a subclass of Exception. ThreadUtils.awaitResult
            // will catch this exception and re-throw the wrapped fatal throwable.
            case oe: OutOfMemoryError =>
              val ex = new Exception(
                new OutOfMemoryError("Not enough memory to build and broadcast the table to all " +
                    "worker nodes. As a workaround, you can either disable broadcast by setting " +
                    s"${SQLConf.AUTO_BROADCASTJOIN_THRESHOLD.key} to -1 or increase the spark " +
                    s"driver memory by setting ${SparkLauncher.DRIVER_MEMORY} to a higher value.")
                    .initCause(oe.getCause))
              promise.failure(ex)
              throw ex
            case e if !NonFatal(e) =>
              val ex = new Exception(e)
              promise.failure(ex)
              throw ex
            case e: Throwable =>
              promise.failure(e)
              throw e
          } finally {
            totalRange.close()
          }
        }
      }
    }
    GpuBroadcastExchangeExec.executionContext.submit[Broadcast[Any]](task)
  }

  override protected[sql] def doExecuteBroadcast[T](): Broadcast[T] = {

    val b = child match {
      case ReusedExchangeExec(_, b: GpuBroadcastExchangeExecBase) => b
      case b: GpuBroadcastExchangeExecBase => b
      case _ => throw new IllegalStateException()
    }

    val x = b.executeColumnarBroadcast()

    throw new UnsupportedOperationException()
  }

  override protected def doExecute(): RDD[InternalRow] = {
    throw new UnsupportedOperationException()
  }
}