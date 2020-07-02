package org.apache.spark.sql.rapids.execution

import org.apache.spark.MapOutputStatistics

class RapidsMapOutputStatistics(shuffleId: Int, bytesByPartitionId: Array[Long])
    extends MapOutputStatistics(shuffleId, bytesByPartitionId)
