"""CLI 入口：python -m image_selection --config image_selection/config.yaml。"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from .configuration import load_config
from .pipeline import run


def main():
    parser=argparse.ArgumentParser(description='JDP Stage1/Stage2 Normal Image Selection')
    parser.add_argument('--config',default='image_selection/config.yaml',help='YAML 設定檔路徑')
    args=parser.parse_args()
    try:
        conf=load_config(args.config)
        result=run(conf)
        return 0 if result['status']=='completed' else 1
    except Exception as exc:
        print(f'[Selection configuration/initialization failed] {type(exc).__name__}: {exc}',file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
