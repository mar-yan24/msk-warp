"""Entry point for SHAC training."""

import argparse
import os
import sys

import yaml

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import warp as wp
wp.init()

from msk_warp.algorithms.shac import SHAC


def main():
    parser = argparse.ArgumentParser(description='SHAC Training')
    parser.add_argument('--cfg', type=str, default='configs/cartpole_shac.yaml')
    parser.add_argument('--logdir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    # Load config
    cfg_path = args.cfg
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(project_root, cfg_path)

    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Override from command line
    if args.logdir is not None:
        cfg['params']['general']['logdir'] = args.logdir
    if args.seed is not None:
        cfg['params']['general']['seed'] = args.seed
    if args.device is not None:
        cfg['params']['general']['device'] = args.device

    shac = SHAC(cfg)
    shac.train()


if __name__ == '__main__':
    main()
