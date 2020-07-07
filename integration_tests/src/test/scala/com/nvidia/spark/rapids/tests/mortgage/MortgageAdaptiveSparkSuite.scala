package com.nvidia.spark.rapids.tests.mortgage

class MortgageAdaptiveSparkSuite extends MortgageSparkSuite {
  override def adaptiveQueryEnabled: Boolean = true
}
