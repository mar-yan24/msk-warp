"""Entry point for PPO training."""

import argparse
import os

import yaml

import warp as wp
wp.init()

from msk_warp import PACKAGE_ROOT
from msk_warp.algorithms.ppo import PPO


def main():
    parser = argparse.ArgumentParser(description='PPO Training')
    parser.add_argument('--cfg', type=str, default='configs/myoleg26_ppo.yaml')
    parser.add_argument('--logdir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    # Resolve config path: check package first, then CWD
    cfg_path = args.cfg
    if not os.path.isabs(cfg_path):
        pkg_path = PACKAGE_ROOT / cfg_path
        if pkg_path.exists():
            cfg_path = str(pkg_path)

    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Override from command line
    if args.logdir is not None:
        cfg['params']['general']['logdir'] = args.logdir
    if args.seed is not None:
        cfg['params']['general']['seed'] = args.seed
    if args.device is not None:
        cfg['params']['general']['device'] = args.device

    ppo = PPO(cfg)
    ppo.train()


if __name__ == '__main__':
    main()
