import os
import sys

# wmcore.py etc. live in cli/ and import each other with plain top-level
# imports (`import wmcore`), not package-relative ones - see wmcore.py's
# module docstring for why (multiprocessing picklability). Tests need the
# same sys.path setup a `python3 cli/watermark.py ...` invocation gets for
# free (Python adds the script's own directory to sys.path).
CLI_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cli"))
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)
