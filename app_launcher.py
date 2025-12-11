#!/usr/bin/env python3
"""Entry point for PyInstaller bundle."""

import sys
import os

# Add the src directory to path for imports
if getattr(sys, 'frozen', False):
    # Running as compiled
    bundle_dir = sys._MEIPASS
    sys.path.insert(0, bundle_dir)
else:
    # Running as script
    bundle_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(bundle_dir, 'src'))

from wtf_am_i_doing.main import main

if __name__ == '__main__':
    main()
