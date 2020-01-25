#!/usr/bin/env python
import argparse
import commands

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Run CLUE algorithm with several parameters.')
  parser.add_argument('-f', '--file',
      action = 'store',
      type = str,
      help = 'File to be processed',
      default = "aniso_1000")
  parser.add_argument('-d', '--dc',
      action = 'store',
      type = float,
      help = "Critical Distance for local density",
      default = 20)
  parser.add_argument('-o', '--deltao',
      action = 'store',
      type = float,
      help = "Critical Distance for outliers",
      default = 20)
  parser.add_argument('-c', '--deltac',
      action = 'store',
      type = float,
      help = "Critical Distance for seeds",
      default = 20)
  parser.add_argument('-r', '--rhoc',
      action = 'store',
      type = float,
      help = "Critical Energy for a Seed",
      default = 50)
  parser.add_argument('-g', '--gpu',
      action = 'store',
      type = bool,
      help = "Run CLUE on GPU",
      default = False)
  parser.add_argument('-n', '--events',
      action = 'store',
      type = int,
      help = "Number of events to generate",
      default = 10)
  parser.add_argument('-v', '--verbose',
      action = 'store',
      type = bool,
      help = "Run in verbose mode",
      default = False)
  parser.add_argument('-b', '--binary',
      action = 'store',
      type = str,
      help = "Executable to run",
      default = "main")

  args = parser.parse_args()

  out_ = commands.getoutput("./%s %s %f %f %f %f %d %d %d" % (
    (args.binary,
      args.file,
      args.dc,
      args.deltao,
      args.deltac,
      args.rhoc,
      args.gpu,
      args.events,
      args.verbose)
    ))

  print(out_)
