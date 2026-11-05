"""Allow `python -m outburst`.

This file defines no functions. It calls ``main`` from ``outburst.main`` and exits with that status.
"""

from outburst.main import main

raise SystemExit(main())
