# conftest.py
import subprocess
import pytest
import logging
import os
import sys

logging.basicConfig()

def pytest_addoption(parser):
    parser.addoption(
        "--test-package",
        action="store",
        default=False,
        help="Run packaging tests",
    )

def pytest_configure(config):
    # Set up logging specifically for pytest_configure
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # Prevent logging from propagating to the root logger
    logger.propagate = False
    
    try:
        logger.info('Starting git LFS pull...')
        result = subprocess.run(["git", "lfs", "pull"], check=True, capture_output=True, text=True)
        logger.info(f"Git LFS pull output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.exception(f"Error running git lfs pull some tests may fail: {e.output}")
        if "Not in a Git repository" in e.output:
            logger.error("You are not in a git repository. Please run this command in a git repository.")
        else:
            pytest.exit(f"Error running git lfs pull some tests may fail: {e.output}")