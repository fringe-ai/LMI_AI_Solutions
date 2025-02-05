import pytest
import logging
import sys
import os
import tempfile
import subprocess

# add path to the repo
PATH = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(PATH))))


@pytest.fixture()
def add_root_path(request):
    if request.config.getoption("--test-package") is False:
        sys.path.append(os.path.join(ROOT, 'lmi_utils'))
        sys.path.append(os.path.join(ROOT, 'anomaly_detectors'))
        logger.info(f"Added {ROOT} to sys.path")
    else:
        logger.info("Skipping adding root path to sys.path")

from anomalib_lmi.anomaly_model import AnomalyModel
from ad_core.anomaly_detector import AnomalyDetector


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


DATA_PATH = 'tests/assets/images/nvtec-ad'
MODEL_PATH = 'tests/assets/models/ad/model_v0.pt'
OUTPUT_PATH = 'tests/assets/validation/ad_v0'



def test_model(add_root_path):
    ad = AnomalyModel(MODEL_PATH)
    ad.test(DATA_PATH, OUTPUT_PATH, generate_stats=True,annotate_inputs=True)

def test_model_api(add_root_path):
    ad = AnomalyDetector(dict(framework='anomalib', model_name='patchcore', task='seg', version='v0'), MODEL_PATH)
    ad.test(DATA_PATH, OUTPUT_PATH, generate_stats=True,annotate_inputs=True)
        
def test_cmds(add_root_path):
    with tempfile.TemporaryDirectory() as t:
        my_env = os.environ.copy()
        my_env['PYTHONPATH'] = f'$PYTHONPATH:{ROOT}/lmi_utils:{ROOT}/anomaly_detectors'
        cmd = f'python -m anomalib_lmi.anomaly_model -i {MODEL_PATH} -d {DATA_PATH} -o {str(t)} -g -p'
        logger.info(f'running cmd: {cmd}')
        result = subprocess.run(cmd,shell=True,env=my_env,capture_output=True,text=True)
        logger.info(result.stdout)
        logger.info(result.stderr)
        
        cmd = f'python -m anomalib_lmi.anomaly_model -a convert -i {MODEL_PATH} -e {str(t)}'
        logger.info(f'running cmd: {cmd}')
        result = subprocess.run(cmd,shell=True,env=my_env,capture_output=True,text=True)
        logger.info(result.stdout)
        logger.info(result.stderr)
