#!/usr/bin/env python
"""Batch runner for all model training scripts."""

import os
import subprocess
import sys

# 获取当前脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 模型训练脚本列表
TRAIN_SCRIPTS = [
    'xgboost_train.py',
    'lightgbm_train.py',
    'rf_train.py',
    'catboost_train.py',
    'ann_train.py',
    'svm_train.py',
    'knn_train.py',
    'nb_train.py',
]

def run_script(script_name):
    """Run a single training script."""
    script_path = os.path.join(SCRIPT_DIR, script_name)
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"{'='*60}\n")

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=PROJECT_ROOT,
        env={**os.environ, 'PYTHONPATH': PROJECT_ROOT}
    )

    if result.returncode != 0:
        print(f"[ERROR] {script_name} failed with exit code {result.returncode}")
        return False
    print(f"[OK] {script_name} completed successfully")
    return True

def main():
    """Run all training scripts."""
    # 清空 performance.csv
    perf_csv = os.path.join(PROJECT_ROOT, 'tables', 'performance.csv')
    if os.path.exists(perf_csv):
        os.remove(perf_csv)
        print(f"Cleared existing {perf_csv}")

    print(f"Starting batch training for all models...")
    print(f"Project root: {PROJECT_ROOT}")

    success_count = 0
    fail_count = 0

    for script in TRAIN_SCRIPTS:
        if run_script(script):
            success_count += 1
        else:
            fail_count += 1

    print(f"\n{'='*60}")
    print(f"Batch training completed!")
    print(f"Success: {success_count}/{len(TRAIN_SCRIPTS)}")
    print(f"Failed: {fail_count}/{len(TRAIN_SCRIPTS)}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
