"""Package executable entrypoint for python -m pcb2bom."""

import sys
from pcb2bom.cli import main

if __name__ == "__main__":
    sys.exit(main())
