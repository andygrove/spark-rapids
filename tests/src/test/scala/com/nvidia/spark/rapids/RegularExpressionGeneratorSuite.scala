package com.nvidia.spark.rapids

import scala.util.Random

import org.scalatest.funsuite.AnyFunSuite

class RegularExpressionGeneratorSuite extends AnyFunSuite {

  class RegexpGen(r: Random) {

    def generate(pattern: RegexAST): String = {
      pattern match {
        case RegexSequence(parts) =>
          parts.flatMap(p => generate(p)).mkString("")
        case RegexRepetition(ast, QuantifierFixedLength(n)) =>
          Range(0, n).map(_ => generate(ast)).mkString("")
        case RegexCharacterClass(false, characters) =>
          generate(characters(r.nextInt(characters.length)))
        case RegexCharacterRange(RegexChar(start), RegexChar(end)) =>
          (start.toInt + r.nextInt(end.toInt - start.toInt)).toChar.toString
        case RegexChar(ch) =>
          ch.toString
        case RegexEscaped(ch) =>
          ch.toString
        case RegexGroup(_, term, _) =>
          generate(term)
        case other =>
          throw new UnsupportedOperationException(other.getClass.getName + ":" + other)
      }
    }
  }



  test("generate string from regexp pattern") {
//    println(doGen("[0-9]{4}-[0-9]{2}-[0-9]{2}"))
    println(doGen("\\[ \\{ [a-z]{3}(, [0-9]{2}){3} \\} \\]"))
  }

  private def doGen(pattern: String): String = {
    val parser = new RegexParser(pattern)
    val ast = parser.parse()
    val gen = new RegexpGen(new Random(42))
    gen.generate(ast)
  }

}
