"""
env_setup.py

Instantiate the environment.
"""

import os

# create the directories
if not os.path.exists("saved_models"):
    os.mkdir("saved_models")

if not os.path.exists("output"):
    os.mkdir("output")
