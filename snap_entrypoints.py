import runpy
import os
import sys

# This gives you the directory where snap_entrypoints.py lives
_BASE = os.path.dirname(os.path.abspath(__file__))
_UQ = os.path.join(_BASE, "uncertainty_quantification")

def run_embeddings_mace():
    if _UQ not in sys.path:
        sys.path.insert(0, _UQ)
    runpy.run_path(os.path.join(_BASE, "uncertainty_quantification/run_embeddings_mace.py"), run_name="__main__")

def run_gbm():
    if _UQ not in sys.path:
        sys.path.insert(0, _UQ)
    runpy.run_path(os.path.join(_BASE, "uncertainty_quantification/run-gbm.py"), run_name="__main__")

def run_embeddings_uma():
    if _UQ not in sys.path:
        sys.path.insert(0, _UQ)
    runpy.run_path(os.path.join(_BASE, "uncertainty_quantification/run_embeddings_uma.py"), run_name="__main__")

def train_gbm():
    if _UQ not in sys.path:
        sys.path.insert(0, _UQ)
    runpy.run_path(os.path.join(_BASE, "uncertainty_quantification/train-gbm.py"), run_name="__main__")

def quantile_prediction():
    if _UQ not in sys.path:
        sys.path.insert(0, _UQ)
    runpy.run_path(os.path.join(_BASE, "uncertainty_quantification/quantile-prediction.py"), run_name="__main__")

def structure_selection():
    if _UQ not in sys.path:
        sys.path.insert(0, _UQ)
    runpy.run_path(os.path.join(_BASE, "uncertainty_quantification/structure-selection.py"), run_name="__main__")

def screen_sites():
    if _UQ not in sys.path:
        sys.path.insert(0, _UQ)
    runpy.run_path(os.path.join(_BASE, "uncertainty_quantification/screen_sites.py"), run_name="__main__")
