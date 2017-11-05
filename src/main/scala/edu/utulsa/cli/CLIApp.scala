package edu.utulsa.cli

trait CLIApp extends App {
  protected implicit val $: CLIParser = CLIParser.parse(args)
}
