"""NourAI CLI shim; corpus operations are delegated to the Phase 8 package."""
import argparse
import sys

from corpus.__main__ import main as corpus_main


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] != "corpus":
        parser = argparse.ArgumentParser(prog="python -m nourai")
        parser.add_argument("command", choices=("corpus",), help="run NourAI corpus tools")
        parser.print_help()
        return 2 if not argv else 2
    return corpus_main(argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
