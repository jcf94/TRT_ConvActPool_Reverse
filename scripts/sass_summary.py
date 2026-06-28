#!/usr/bin/env python3
import argparse
import collections
import re
import subprocess
from pathlib import Path


FAMILIES = (
    "IMMA",
    "DP4A",
    "HMMA",
    "F2IP",
    "I2FP",
    "FFMA",
    "LDG",
    "LDS",
    "STS",
    "STG",
    "BAR",
)


def read_sass(args):
  path = Path(args.input)
  if args.sass:
    return path.read_text(errors="ignore")
  cmd = [args.cuobjdump, "-sass"]
  if args.arch:
    cmd += ["-arch", args.arch]
  cmd.append(str(path))
  return subprocess.check_output(cmd, text=True, errors="ignore")


def summarize_functions(sass):
  summaries = collections.OrderedDict()
  current = "GLOBAL"
  summaries[current] = collections.Counter()
  for line in sass.splitlines():
    match = re.search(r"Function\s*:\s*(.*)", line)
    if match:
      current = match.group(1).strip()
      summaries[current] = collections.Counter()
      continue
    for family in FAMILIES:
      if family in line:
        summaries[current][family] += 1
  return summaries


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("input", help="SASS text or CUDA binary/fatbin")
  parser.add_argument("--sass", action="store_true",
                      help="treat input as existing SASS text")
  parser.add_argument("--arch", default="sm_86")
  parser.add_argument("--cuobjdump", default="cuobjdump")
  args = parser.parse_args()

  summaries = summarize_functions(read_sass(args))
  print("function," + ",".join(FAMILIES))
  for name, counts in summaries.items():
    if not any(counts.values()):
      continue
    values = [str(counts.get(family, 0)) for family in FAMILIES]
    print(f"{name}," + ",".join(values))


if __name__ == "__main__":
  main()
