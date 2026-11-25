"""每產品資料夾、圖片可讀性檢查，並保障原始 Train Set 不被修改。"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
from pathlib import Path
from PIL import Image

EXTENSIONS = {'.png','.jpg','.jpeg','.bmp'}


@dataclass(frozen=True)
class ImageItem:
    path: Path
    part: str
    source: str  # 'train' / 'candidate' / 'test' / 'val'

    @property
    def key(self) -> str:
        return str(self.path.resolve())

    @property
    def name(self) -> str:
        return self.path.name


def find_images(folder: Path, recursive: bool) -> list[Path]:
    if not folder.is_dir():
        return []
    it = folder.rglob('*') if recursive else folder.iterdir()
    return sorted((p.resolve() for p in it if p.is_file() and p.suffix.lower() in EXTENSIONS), key=str)


def _file_hash(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda:fh.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def check_image(path: Path) -> tuple[bool,str]:
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.convert('RGB').load()
        return True, ''
    except Exception as exc:
        return False, f'{type(exc).__name__}: {exc}'


def discover_parts(conf: dict, root: Path) -> list[str]:
    configured = conf['dataset']['parts']
    if configured == 'all':
        return sorted(p.name for p in root.iterdir() if p.is_dir() and
                      (p / conf['dataset']['train_rel']).is_dir())
    return list(configured)


def discover_data(conf: dict, root: Path, part: str) -> tuple[list[ImageItem],list[ImageItem],list[dict]]:
    ds = conf['dataset']
    train_dir = root / part / ds['train_rel']
    candidate_dir = root / part / ds['candidate_rel']
    if not train_dir.is_dir():
        raise FileNotFoundError(f'{part}: 沒有原始 train/good：{train_dir}')
    if not candidate_dir.is_dir():
        raise FileNotFoundError(f'{part}: 沒有獨立候選資料夾：{candidate_dir}')
    train = [ImageItem(p,part,'train') for p in find_images(train_dir, ds['recursive'])]
    candidate = [ImageItem(p,part,'candidate') for p in find_images(candidate_dir, ds['recursive'])]
    skipped = []
    filtered = []
    seen = set()
    train_keys = set()
    seen_content=set()
    for im in train:
        ok,reason = check_image(im.path)
        if not ok:
            skipped.append({'part':part,'source':'train','path':str(im.path),'reason':reason})
            continue
        if im.key in seen:
            continue
        content_hash=_file_hash(im.path)
        if content_hash in seen_content:
            skipped.append({'part':part,'source':'train','path':str(im.path),'reason':'與其他原始正常圖片檔案內容完全重複'})
            continue
        seen_content.add(content_hash)
        filtered.append(im)
        seen.add(im.key)
        train_keys.add(im.key)
    train = filtered
    filtered = []
    for im in candidate:
        ok,reason = check_image(im.path)
        if not ok:
            skipped.append({'part':part,'source':'candidate','path':str(im.path),'reason':reason})
            continue
        if im.key in seen or im.key in train_keys:
            skipped.append({'part':part,'source':'candidate','path':str(im.path),'reason':'與訓練圖片檔案路徑重複'})
            continue
        content_hash=_file_hash(im.path)
        if content_hash in seen_content:
            skipped.append({'part':part,'source':'candidate','path':str(im.path),'reason':'與其他 train/candidate 圖片檔案內容完全重複'})
            continue
        seen_content.add(content_hash)
        filtered.append(im)
        seen.add(im.key)
    if not train:
        raise ValueError(f'{part}: 沒有任何有效的原始 Training Images，不能建立 Stage 2 PCA。')
    return train,filtered,skipped
