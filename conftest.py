# conftest.py
import subprocess
import pytest
import logging
import sys

logging.basicConfig()

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
        logger.info(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.exception(f"Error running git lfs pull: {e.stderr}")
        pytest.exit("Exiting tests due to git lfs pull failure")
