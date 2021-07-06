package com.nvidia.spark.rapids

import ai.rapids.cudf.{ColumnView, DType, HostColumnVector, HostColumnVectorCore}
import org.junit.jupiter.api.Assertions.{assertArrayEquals, assertEquals}

/**
 * Convenience methods for testing cuDF calls directly
 */
object CudfTestHelper {

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
        case BOOL8 | INT8 | UINT8 =>
          assertEquals(expected.getByte(expectedRow), cv.getByte(tableRow),
            "Column " + colName + " Row " + tableRow)

        case INT16 | UINT16 =>
          assertEquals(expected.getShort(expectedRow), cv.getShort(tableRow),
            "Column " + colName + " Row " + tableRow)

        case INT32 | UINT32 | TIMESTAMP_DAYS | DURATION_DAYS | DECIMAL32 =>
          assertEquals(expected.getInt(expectedRow), cv.getInt(tableRow),
            "Column " + colName + " Row " + tableRow)

        case INT64 | UINT64 |
             DURATION_MICROSECONDS | DURATION_MILLISECONDS | DURATION_NANOSECONDS |
             DURATION_SECONDS | TIMESTAMP_MICROSECONDS | TIMESTAMP_MILLISECONDS |
             TIMESTAMP_NANOSECONDS | TIMESTAMP_SECONDS | DECIMAL64 =>
          assertEquals(expected.getLong(expectedRow), cv.getLong(tableRow),
            "Column " + colName + " Row " + tableRow)

        case STRING =>
          assertArrayEquals(expected.getUTF8(expectedRow), cv.getUTF8(tableRow),
            "Column " + colName + " Row " + tableRow)

        case _ =>
          throw new IllegalArgumentException(`type` + " is not supported yet")
      }
    }
  }


}
