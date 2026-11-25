"""本地測試輔助：複製 MVTec train/good 成獨立的 train/candidates，絕不修改來源。"""
from __future__ import annotations
import argparse
import random
from pathlib import Path
import shutil

ALLOWED={'.png','.jpg','.jpeg','.bmp'}


def main():
    parser=argparse.ArgumentParser(description='MVTec normal-image split for selection smoke test')
    parser.add_argument('--source',required=True,help='MVTec 原始根目錄，例如 datasets/mvtec-ad')
    parser.add_argument('--destination',required=True,help='新測試資料根目錄；與 source 必須不同')
    parser.add_argument('--part',default='bottle',help='欲測試的產品類別')
    parser.add_argument('--initial',type=int,default=3,help='原始 Training Set 保留幾張')
    parser.add_argument('--candidates',type=int,default=10,help='獨立候選資料夾複製幾張')
    parser.add_argument('--seed',type=int,default=42,help='固定分割亂數')
    args=parser.parse_args()
    src=Path(args.source).expanduser().resolve()
    dst=Path(args.destination).expanduser().resolve()
    if src==dst or src in dst.parents or dst in src.parents:
        parser.error('source 與 destination 不可相同或互為上/下層，避免覆寫來源資料。')
    inp=src/args.part/'train'/'good'
    photos=sorted(p for p in inp.iterdir() if p.suffix.lower() in ALLOWED and p.is_file())
    total=args.initial+args.candidates
    if args.initial<1 or args.candidates<1 or total>len(photos):
        parser.error(f'至少需要 {total} 張圖片，來源實際 {len(photos)} 張。')
    if (dst/args.part).exists():
        parser.error('目的產品資料夾已存在；請換一個 destination，避免覆寫。')
    chosen=random.Random(args.seed).sample(photos,total)
    for im in chosen[:args.initial]:
        dest=dst/args.part/'train'/'good'/im.name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(im,dest)
    for im in chosen[args.initial:]:
        dest=dst/args.part/'candidates'/'good'/im.name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(im,dest)
    print(f'輸出完成：{dst/args.part}；train={args.initial}，candidates={args.candidates}，seed={args.seed}')


if __name__=='__main__':
    main()
