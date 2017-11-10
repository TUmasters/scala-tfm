package edu.utulsa.cli

trait CLIApp extends App {
  private var parser: CLIParser = _
  protected implicit var tree: ParamTree = _

  // Will be deprecated in the near future; see https://issues.scala-lang.org/browse/SI-4330
  delayedInit {
    parser = CLIParser.parse(args)
    tree = parser.tree
  }

  def $: CLIParser = parser
}
