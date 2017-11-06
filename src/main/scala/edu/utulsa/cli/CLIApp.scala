package edu.utulsa.cli

trait CLIApp extends App {
  protected implicit var parser: CLIParser = _
  protected implicit var tree: ParamTree = _
  delayedInit {
    parser = CLIParser.parse(args)
    tree = parser.tree
  }

  def $: CLIParser = parser
}
