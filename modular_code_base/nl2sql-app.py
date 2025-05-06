#!/usr/bin/env python3
"""
Natural Language to SQL Query Application
----------------------------------------
A modular application that converts natural language questions 
into SQL queries using OpenAI's language models.
"""

import sys
import os

# Add the parent directory to Python path for module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the main module
from nl2sql.main import main

if __name__ == "__main__":
    main()