import os
import sys

PROJECT_PATH = os.getcwd()
SOURCE_PATH = os.path.join(
    PROJECT_PATH, "inference"  # important for imports
)
sys.path.append(SOURCE_PATH)


