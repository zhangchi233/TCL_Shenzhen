#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
import multiprocessing as mp
from typing import Any, Dict, List, Optional, Tuple, Set
import shutil
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# ============================================================================
# Purpose
# ----------------------------------------------------------------------------
# This script merges multiple extracted "visual" crops (image/chart) that belong
# to the same multi-panel figure into one composite image.
#
# It works per PDF page JSON produced by layout+parsing pipeline:
#   - layout_det_res.boxes: provides (label, coordinate)
#   - parsing_res_list: provides (block_id, block_bbox, block_content)
#
# Key idea:
#   1) Collect relevant blocks (visuals + figure_title)
#   2) Detect which figure_title blocks are "captions" vs "subfigure subtitles"
#   3) Assign each visual to the nearest caption (if within threshold)
#   4) Cluster visuals using explicit merging rules (caption-aware + geometry)
#   5) Render each cluster by pasting member crops + drawing subtitles
#
# All hyperparameters are configurable via CLI.
# ============================================================================


# ============================================================================
# 1) filename bbox parsing for extracted crops
# ----------------------------------------------------------------------------
# expects ..._<x1>_<y1>_<x2>_<y2>.<ext>
# e.g. img_in_image_box_342_460_744_735.jpg
# ============================================================================
BBOX_RE = re.compile(r"_(\d+)_(\d+)_(\d+)_(\d+)\.(jpg|jpeg|png|webp)$", re.IGNORECASE)


def parse_bbox_from_filename(name: str) -> Optional[Tuple[int, int, int, int]]:
    m = BBOX_RE.search(name)
    if not m:
        return None
    x1, y1, x2, y2 = map(int, m.groups()[:4])
    if x1 >= x2 or y1 >= y2:
        return None
    return (x1, y1, x2, y2)


def label_hint_from_filename(name: str) -> Optional[str]:
    """
    Best-effort label hint from crop filename.
    Used only when multiple crops share the same bbox (rare) and we want to
    select the best candidate consistent with the visual label.
    """
    n = name.lower()
    if "img_in_chart_box_" in n or "_chart_box_" in n:
        return "chart"
    if "img_in_image_box_" in n or "_image_box_" in n:
        return "image"
    return None


# ============================================================================
# 2) caption / subtitle detection
# ----------------------------------------------------------------------------
# - "caption title": usually starts with Fig/Figure/图/表/Table...
# - "subfigure subtitle": (a) / (b) / （a） / (1) / (i) ... (optionally with text)
# ============================================================================
_CAPTION_PREFIX_RE = re.compile(
    r"^\s*(Fig\.?|FIG\.?|Figure|图|圖|表|Table|TABLE)\b",
    re.IGNORECASE,
)

_PANEL_PREFIX_RE = re.compile(
    r"^\s*[\(（]\s*([A-Za-z]|[0-9]+|[ivxlcdmIVXLCDM]{1,6})\s*[\)）]"
)


def is_caption_title(text: str) -> bool:
    return _CAPTION_PREFIX_RE.match((text or "").strip()) is not None


def is_subfigure_subtitle(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if is_caption_title(t):
        return False
    return _PANEL_PREFIX_RE.match(t) is not None


# ============================================================================
# 3) block model
# ============================================================================
@dataclass(frozen=True)
class Block:
    block_id: int
    label: str
    bbox: Tuple[int, int, int, int]
    content: str = ""

    @property
    def x1(self) -> int: return self.bbox[0]
    @property
    def y1(self) -> int: return self.bbox[1]
    @property
    def x2(self) -> int: return self.bbox[2]
    @property
    def y2(self) -> int: return self.bbox[3]

    @property
    def is_visual(self) -> bool:
        return self.label in {"image", "chart"}

    @property
    def is_title(self) -> bool:
        return self.label == "figure_title"


# ============================================================================
# 4) geometry utilities
# ============================================================================
def expand_box(b: Tuple[int, int, int, int], margin: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = b
    return (x1 - margin, y1, x2 + margin, y2 + margin) # do not expand upward since figure subtitles are always below 
                                                        # (keep only y1, instead of y1 - margin)


def boxes_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def overlap_1d(a1: int, a2: int, b1: int, b2: int) -> int:
    """Length of overlap between [a1,a2] and [b1,b2]."""
    return max(0, min(a2, b2) - max(a1, b1))


def gap_1d(a1: int, a2: int, b1: int, b2: int) -> int:
    """Distance between intervals (0 if overlapping)."""
    if a2 < b1:
        return b1 - a2
    if b2 < a1:
        return a1 - b2
    return 0


def are_adjacent_bbox(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
    max_gap: int,
    min_overlap: int = 1,
) -> bool:
    """
    Axis-adjacency test: merge only when aligned on one axis and close on the other.

    - Vertical adjacency:
        x-overlap >= min_overlap AND vertical gap <= max_gap
    - Horizontal adjacency:
        y-overlap >= min_overlap AND horizontal gap <= max_gap
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    x_ov = overlap_1d(ax1, ax2, bx1, bx2)
    y_ov = overlap_1d(ay1, ay2, by1, by2)

    x_gap = gap_1d(ax1, ax2, bx1, bx2)
    y_gap = gap_1d(ay1, ay2, by1, by2)

    vertical_adj = (x_ov >= min_overlap) and (y_gap <= max_gap)
    horizontal_adj = (y_ov >= min_overlap) and (x_gap <= max_gap)
    return vertical_adj or horizontal_adj


def union_bbox(blocks: List[Block]) -> Tuple[int, int, int, int]:
    return (
        # min(b.x1 for b in blocks),
        # min(b.y1 for b in blocks),
        # max(b.x2 for b in blocks),
        # max(b.y2 for b in blocks),
        blocks[0].x1,
        blocks[0].y1,
        blocks[0].x2,
        blocks[0].y2,
    )


def bbox_bottom_center(b: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) /2.0)


def bbox_edge_distance(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """
    Euclidean distance between two axis-aligned bboxes (0 if overlap).
    This is a robust "how close are the rectangles" metric even when not aligned.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    if ax2 < bx1:
        dx = bx1 - ax2
    elif bx2 < ax1:
        dx = ax1 - bx2
    else:
        dx = 0

    if ay2 < by1:
        dy = by1 - ay2
    elif by2 < ay1:
        dy = ay1 - by2
    else:
        dy = 0

    return (dx * dx + dy * dy) ** 0.5


def point_to_bbox_distance(px: float, py: float, b: Tuple[int, int, int, int]) -> float:
    """Euclidean distance from point to bbox (0 if point is inside bbox)."""
    x1, y1, x2, y2 = b
    dx = 0.0
    if px < x1:
        dx = x1 - px
    elif px > x2:
        dx = px - x2
    dy = 0.0
    if py < y1:
        dy = y1 - py
    elif py > y2: # caption is above visual - automaticly reject it, since caption should be below visual
        #dy = py - y2
        dy = 999999
    return (dx * dx + dy * dy) ** 0.5


# ============================================================================
# 5) caption assignment
# ----------------------------------------------------------------------------
# For each visual crop, we assign the nearest caption (figure_title recognized as
# caption) if it is within MAX_CAPTION_DIST pixels.
#
# ============================================================================
def assign_caption_to_visuals(
    visuals: List[Block],
    captions: List[Block],
    max_caption_dist: int,
) -> Dict[int, int]:
    """
    Returns mapping: visual_index -> caption_block_id

    Rule:
      - Each visual takes its nearest caption if distance <= max_caption_dist.
      - If no caption is close enough, the visual is treated as "no caption".
    """
    out: Dict[int, int] = {}
    if not captions:
        return out

    caption_boxes = [(c.block_id, c.bbox) for c in captions]

    for vi, v in enumerate(visuals):
        cx, cy = bbox_bottom_center(v.bbox)

        best_id: Optional[int] = None
        best_d: Optional[float] = None
        for cid, cbb in caption_boxes:
            d = point_to_bbox_distance(cx, cy, cbb)
            if best_d is None or d < best_d:
                best_d = d
                best_id = cid
        if best_id is not None and best_d is not None and best_d <= max_caption_dist:
            out[vi] = best_id
    return out


# ============================================================================
# 6) DSU for clustering
# ============================================================================
class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1


# ============================================================================
# 7) CLUSTERING: merging rules + hyperparameters
# ----------------------------------------------------------------------------
# Visual clustering is the core of the script.
#
# We merge two visuals i and j if ANY of the allowed rules fire, subject to
# hard constraints:
#
# HARD CONSTRAINT:
#   (A) If both visuals have captions AND captions differ => NEVER MERGE.
#
# OTHERWISE:
#   (B) If both visuals have the SAME caption:
#               always merge
#       (because panels in one figure can be far apart / not aligned).
#
#   (C) If one or both visuals have no caption:
#       Merge conservatively using geometry:
#         C1) axis-adjacent: are_adjacent_bbox(...) with max_gap + min_overlap
#         OR
#         C2) close-by even if not aligned: bbox_edge_distance <= no_caption_max_dist
# ============================================================================
def cluster_visuals_with_caption_rule(
    visuals: List[Block],
    visual_to_caption: Dict[int, int],
    *,
    no_caption_max_gap: int,
    min_overlap: int,
    no_caption_max_dist: int,
) -> List[List[Block]]:
    n = len(visuals)
    if n == 0:
        return []

    dsu = DSU(n)

    for i in range(n):
        for j in range(i + 1, n):
            ci = visual_to_caption.get(i)
            cj = visual_to_caption.get(j)

            # (A) Different captions => never merge
            if ci is not None and cj is not None and ci != cj:
                print(f' Rule A, cj {cj} and ci {ci}')
                continue

            a = visuals[i].bbox
            b = visuals[j].bbox

            # (B) Same caption => ALWAYS merge
            if ci is not None and cj is not None and ci == cj:
                print(f' Rule B, cj {cj} and ci {ci}')
                dsu.union(i, j)
                continue

            #(C) Missing caption on at least one side => conservative merge
            if are_adjacent_bbox(a, b, max_gap=no_caption_max_gap, min_overlap=min_overlap):
                print(f' Rule C, cj {cj} and ci {ci}')
                dsu.union(i, j)
                continue

            if bbox_edge_distance(a, b) <= no_caption_max_dist:
                dsu.union(i, j)
                continue

    groups: Dict[int, List[Block]] = {}
    for i, b in enumerate(visuals):
        groups.setdefault(dsu.find(i), []).append(b)

    clusters = list(groups.values())
    clusters.sort(key=lambda g: (min(b.y1 for b in g), min(b.x1 for b in g)))
    for g in clusters:
        g.sort(key=lambda b: (b.y1, b.x1))
    return clusters


# ============================================================================
# 8) image index for crops
# ============================================================================
def build_image_index(img_root: Path) -> Dict[Tuple[int, int, int, int], List[Path]]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    idx: Dict[Tuple[int, int, int, int], List[Path]] = {}
    for p in img_root.rglob("*"):
        if p.suffix.lower() not in exts:
            continue
        bb = parse_bbox_from_filename(p.name)
        if bb is None:
            continue
        idx.setdefault(bb, []).append(p)
    return idx


def pick_best_path(candidates: List[Path], label: str) -> Path:
    if len(candidates) == 1:
        return candidates[0]
    preferred = [p for p in candidates if (label_hint_from_filename(p.name) == label)]
    if preferred:
        preferred.sort(key=lambda p: (len(str(p)), str(p)))
        return preferred[0]
    candidates.sort(key=lambda p: (len(str(p)), str(p)))
    return candidates[0]


# ============================================================================
# 9) font loading
# ============================================================================
def load_font(font_path: Optional[str], font_size: int) -> ImageFont.ImageFont:
    if font_path:
        return ImageFont.truetype(font_path, font_size)

    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/truetype/arphic/ukai.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/System/Library/Fonts/PingFang.ttc",  # macOS
        "/System/Library/Fonts/STHeiti Light.ttc",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, font_size)

    try:
        return ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        return ImageFont.load_default()


# ============================================================================
# 10) rendering: paste member crops + draw subfigure subtitles
# ----------------------------------------------------------------------------
# Saves only if >= min_visuals_to_save visuals were actually pasted.
# Also writes sidecar *.missing.txt listing missing crop bboxes.
# ============================================================================
def render_cluster(
    cluster_visuals: List[Block],
    cluster_titles: List[Block],
    bbox_to_paths: Dict[Tuple[int, int, int, int], List[Path]],
    out_path: Path,
    *,
    pad: int,
    font: ImageFont.ImageFont,
    background: Tuple[int, int, int],
    min_visuals_to_save: int,
    subtitle_inset_x: int,
    subtitle_inset_y: int,
    subtitle_bg_pad: int,
    write_missing_sidecar: bool,
    verbose: bool,
) -> int:
    if len(cluster_visuals) < min_visuals_to_save:
        return 0

    subtitles = [t for t in cluster_titles if is_subfigure_subtitle(t.content)]
    all_blocks = list(cluster_visuals) + subtitles

    ux1, uy1, ux2, uy2 = union_bbox(all_blocks)
    W = (ux2 - ux1) + 2 * pad
    H = (uy2 - uy1) + 2 * pad

    canvas = Image.new("RGB", (W, H), background)
    draw = ImageDraw.Draw(canvas)

    pasted = 0
    missing_visuals: List[Block] = []

    for b in cluster_visuals:
        cand = bbox_to_paths.get(b.bbox)
        if not cand:
            missing_visuals.append(b)
            continue
        img_path = pick_best_path(cand, b.label)
        img = Image.open(img_path).convert("RGB")

        px = (b.x1 - ux1) + pad
        py = (b.y1 - uy1) + pad
        canvas.paste(img, (px, py))
        pasted += 1

    # draw subtitles with background box
    for t in subtitles:
        s = t.content.strip()
        if not s:
            continue

        tx = (t.x1 - ux1) + pad + subtitle_inset_x
        ty = (t.y1 - uy1) + pad + subtitle_inset_y

        left, top, right, bottom = draw.textbbox((tx, ty), s, font=font)
        draw.rectangle(
            (left - subtitle_bg_pad, top - subtitle_bg_pad, right + subtitle_bg_pad, bottom + subtitle_bg_pad),
            fill=background,
        )
        draw.text((tx, ty), s, fill=(0, 0, 0), font=font)

    if pasted >= min_visuals_to_save:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path)
        if verbose:
            print(f"[SAVED] {out_path}")

    if write_missing_sidecar and missing_visuals:
        sidecar = out_path.with_suffix(out_path.suffix + ".missing.txt")
        with sidecar.open("w", encoding="utf-8") as f:
            for b in missing_visuals:
                f.write(f"missing visual bbox={b.bbox} label={b.label} block_id={b.block_id}\n")

    return pasted


# ============================================================================
# 11) JSON parsing helpers
# ============================================================================
def find_block_by_bbox(parsing_res_list: list, bbox: List[int]) -> Optional[Tuple[int, str]]:
    for b in parsing_res_list:
        if b.get("block_bbox") == bbox:
            return b.get("block_id"), b.get("block_content")
    return None


def extract_blocks_from_page_json(
    data: Dict[str, Any],
    labels: Set[str],
) -> List[Block]:
    blocks: List[Block] = []
    boxes = data.get("layout_det_res", {}).get("boxes", [])
    parsing_res_list = data.get("parsing_res_list", [])

    for item in boxes:
        label = item.get("label")


        bbox = item.get("coordinate")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        bbox = [int(i) for i in bbox]
        x1, y1, x2, y2 = bbox
        if x1 >= x2 or y1 >= y2:
            continue

        hit = find_block_by_bbox(parsing_res_list, bbox)
        if hit is None:
            continue
        bid, block_content = hit
        if not isinstance(bid, int):
            continue

        blocks.append(
            Block(
                block_id=bid,
                label=label,
                bbox=(x1, y1, x2, y2),
                content=(block_content or "").strip(),
            )
        )
    return blocks


# ============================================================================
# 12) per-page JSON processing
# ============================================================================
def process_json(
    json_path: Path,
    bbox_to_paths: Dict[Tuple[int, int, int, int], List[Path]],
    out_dir: Path,
    labels: Set[str],
    *,
    margin: int,
    pad: int,
    font: ImageFont.ImageFont,
    max_caption_dist: int,
    no_caption_max_gap: int,
    min_overlap: int,
    no_caption_max_dist: int,
    min_visuals_to_save: int,
    background: Tuple[int, int, int],
    subtitle_inset_x: int,
    subtitle_inset_y: int,
    subtitle_bg_pad: int,
    write_missing_sidecar: bool,
    verbose: bool,
) -> int:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    pdf_stem = Path(data.get("input_path", "unknown.pdf")).stem
    page_index = int(data.get("page_index", -1))

    blocks = extract_blocks_from_page_json(data, labels=labels)
    visuals = [b for b in blocks if b.is_visual]
    titles = [b for b in blocks if b.is_title]

    # captions (figure titles that look like "Fig. 1 ..." / "图1 ..." etc.)
    captions = [t for t in titles if is_caption_title(t.content)]

   

    visual_to_caption = assign_caption_to_visuals(
        visuals,
        captions,
        max_caption_dist=max_caption_dist,
    )

    # cluster visuals by caption-aware merging rules
    visual_clusters = cluster_visuals_with_caption_rule(
        visuals,
        visual_to_caption,
        no_caption_max_gap=no_caption_max_gap,
        min_overlap=min_overlap,
        no_caption_max_dist=no_caption_max_dist,
    )
    if not visual_clusters:
       
        return 0

    n_saved = 0
    out_cid = 0
    
    for vcluster in visual_clusters:
        if len(vcluster) < min_visuals_to_save:
            continue

        # include any titles/subtitles overlapping the cluster union (with margin)
        cbb = expand_box(union_bbox(vcluster), margin=margin)
        cluster_titles = [t for t in titles if boxes_overlap(t.bbox, cbb)]

        ux1, uy1, ux2, uy2 = union_bbox(vcluster)
        out_name = f"{pdf_stem}_page_{page_index:04d}_cluster_{out_cid:02d}_{ux1}_{uy1}_{ux2}_{uy2}.png"
        out_path = out_dir / out_name

        pasted = render_cluster(
            cluster_visuals=vcluster,
            cluster_titles=cluster_titles,
            bbox_to_paths=bbox_to_paths,
            out_path=out_path,
            pad=pad,
            font=font,
            background=background,
            min_visuals_to_save=min_visuals_to_save,
            subtitle_inset_x=subtitle_inset_x,
            subtitle_inset_y=subtitle_inset_y,
            subtitle_bg_pad=subtitle_bg_pad,
            write_missing_sidecar=write_missing_sidecar,
            verbose=verbose,
        )
        if pasted >= min_visuals_to_save:
            out_cid += 1
            n_saved += 1
    print("saved %d images" % n_saved)
    return n_saved


# ============================================================================
# 13) crawl categories/year/issue/paper and run per paper
# ============================================================================
def _has_any_json(p: Path) -> bool:
    try:
        next(p.rglob("*.json"))
        return True
    except StopIteration:
        return False


def _has_any_bbox_images(p: Path) -> bool:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    for img in p.rglob("*"):
        if img.suffix.lower() not in exts:
            continue
        if parse_bbox_from_filename(img.name) is not None:
            return True
    return False


def find_paper_jobs(
    base_root,
    json_subdir: str = "rec_json",
    img_subdir: str = "imgs",
) -> List[Tuple[Path, Path, Path, Tuple[str, str, str, str]]]:
    """
    Returns list of jobs:
      (paper_dir, json_dir, img_dir, (category, year, issue, paper_name))

    Supported trees:

    (A) Default journals layout:
      base_root/<category>/<year>/<issue>/<paper_name>/<json_subdir>/<...json...>
      base_root/<category>/<year>/<issue>/<paper_name>/<img_subdir>/<...crops...>

    (B) Books layout (category == "中文书籍"):
      base_root/<category>/<paper_name>/<json_subdir>/<...json...>
      base_root/<category>/<paper_name>/<img_subdir>/<...crops...>

    Auto-detect fallback:
      If json_subdir/img_subdir don't exist, pick the first subdir under paper_dir
      that contains JSON / bbox-images.
    """
    jobs = []

    def resolve_dirs(paper_dir: Path) -> Optional[Tuple[Path, Path]]:
        """Return (json_dir, img_dir) for a paper_dir, with fallback autodetect."""
        jdir = paper_dir / json_subdir
        idir = paper_dir / img_subdir

        if not jdir.is_dir():
            candidates = [d for d in paper_dir.iterdir() if d.is_dir() and _has_any_json(d)]
            if candidates:
                jdir = candidates[0]

        if not idir.is_dir():
            candidates = [d for d in paper_dir.iterdir() if d.is_dir() and _has_any_bbox_images(d)]
            if candidates:
                idir = candidates[0]

        if jdir.is_dir() and idir.is_dir() and _has_any_json(jdir) and _has_any_bbox_images(idir):
            return jdir, idir
        return None

    if not isinstance(base_root, Path):
        paper_dir = Path(base_root)
    else:
        paper_dir = base_root
    
        
    paper_name = paper_dir.name
    resolved = resolve_dirs(paper_dir)
            
    jdir, idir = resolved
    
            # year/issue not applicable -> use empty strings to keep tuple shape
    jobs.append((paper_dir, jdir, idir, ("", "", "", paper_name)))


    return jobs


def process_paper(
    json_dir: Path,
    img_dir: Path,
    out_dir: Path,
    labels: Set[str],
    *,
    margin: int,
    pad: int,
    font: ImageFont.ImageFont,
    max_caption_dist: int,
    no_caption_max_gap: int,
    min_overlap: int,
    no_caption_max_dist: int,
    min_visuals_to_save: int,
    background: Tuple[int, int, int],
    subtitle_inset_x: int,
    subtitle_inset_y: int,
    subtitle_bg_pad: int,
    write_missing_sidecar: bool,
    verbose: bool,
) -> int:
    bbox_to_paths = build_image_index(img_dir)
    total_saved = 0
    
    for jp in sorted(json_dir.rglob("*.json")):
        
        total_saved += process_json(
            jp,
            bbox_to_paths,
            out_dir,
            labels,
            margin=margin,
            pad=pad,
            font=font,
            max_caption_dist=max_caption_dist,
            no_caption_max_gap=no_caption_max_gap,
            min_overlap=min_overlap,
            no_caption_max_dist=no_caption_max_dist,
            min_visuals_to_save=min_visuals_to_save,
            background=background,
            subtitle_inset_x=subtitle_inset_x,
            subtitle_inset_y=subtitle_inset_y,
            subtitle_bg_pad=subtitle_bg_pad,
            write_missing_sidecar=write_missing_sidecar,
            verbose=verbose,
        )
    return total_saved


def _process_one_paper(payload):
    job, args_dict = payload
    paper_dir, json_dir, img_dir, (_cat, _year, _issue, paper_name) = job
    out_root = Path(args_dict["out_root"])
   
    paper_out = out_root / paper_name

    try:
        labels = set(args_dict["labels"])
        font = load_font(args_dict["font_path"], args_dict["font_size"])

        n_merged = process_paper(
            json_dir=Path(json_dir),
            img_dir=Path(img_dir),
            out_dir=paper_out,
            labels=labels,
            margin=args_dict["margin"],
            pad=args_dict["pad"],
            font=font,
            max_caption_dist=args_dict["max_caption_dist"],
            no_caption_max_gap=args_dict["no_caption_max_gap"],
            min_overlap=args_dict["min_overlap"],
            no_caption_max_dist=args_dict["no_caption_max_dist"],
            min_visuals_to_save=args_dict["min_visuals_to_save"],
            background=tuple(args_dict["background_rgb"]),
            subtitle_inset_x=args_dict["subtitle_inset_x"],
            subtitle_inset_y=args_dict["subtitle_inset_y"],
            subtitle_bg_pad=args_dict["subtitle_bg_pad"],
            write_missing_sidecar=args_dict["write_missing_sidecar"],
            verbose=args_dict["verbose"],
        )

        return paper_name, {
            "path_to_merged_imgs": str(paper_out.resolve()),
            "path_to_paper_dir": str(Path(paper_dir).resolve()),
            "n_merged_imgs": int(n_merged),
        }

    except Exception as e:
        return paper_name, {"error": str(e), "path_to_paper_dir": str(Path(paper_dir).resolve())}


# ============================================================================
# 14) CLI
# ----------------------------------------------------------------------------
# All hyperparameters are exposed as CLI args.
# ============================================================================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Merge multi-panel extracted visuals (image/chart) into composite clusters (caption-aware)."
    )
    ap.add_argument(
        "base_file",
        help="Root dir containing subdirs of papers, each with <json_subdir> and <img_subdir>.",
    )
  
    ap.add_argument(
        "--categories",
        default="现代显示,液晶与显示,中文书籍",
        help="Comma-separated category folder names (default: 现代显示,液晶与显示)",
    )
    ap.add_argument("--json_subdir", default="rec_json", help="Default subdir name containing page JSON files.")
    ap.add_argument("--img_subdir", default="imgs", help="Default subdir name containing extracted crops.")

    ap.add_argument(
        "--out_root",
        default="/mnt/workspace/MLLM/karol/merge_sub_images/merged_subimages/max_selected",
        help="Root output dir; each paper writes to <out_root>/<paper_name>/...",
    )
    ap.add_argument(
        "--merged_imgs_dict_out",
        default="/mnt/workspace/MLLM/karol/merge_sub_images/merged_imgs_dict.json",
        help="Where to save merged_imgs_dict JSON.",
    )

    # --- block selection ---
    ap.add_argument(
        "--labels",
        default="image,chart,figure_title",
        help="Comma-separated labels to load from layout_det_res (default: image,chart,figure_title).",
    )

    # --- caption assignment hyperparameter ---
    ap.add_argument(
        "--max_caption_dist",
        type=int,
        default=250,
        help=(
            "Max distance (px) to consider a figure_title block as the caption of a visual. "
            "Computed as point-to-bbox distance from visual center to caption bbox."
        ),
    )

    # --- clustering hyperparameters (core) ---
    ap.add_argument(
        "--no_caption_max_gap",
        type=int,
        default=120,
        help=(
            "If one/both visuals have NO caption, allow axis-adjacent merging when gap <= this "
            "(requires overlap on the orthogonal axis)."
        ),
    )
    ap.add_argument(
        "--min_overlap",
        type=int,
        default=1,
        help="Minimum overlap (px) required on the aligned axis for axis-adjacent merging.",
    )
    ap.add_argument(
        "--no_caption_max_dist",
        type=int,
        default=150,
        help=(
            "If one/both visuals have NO caption, also merge if bbox_edge_distance <= this "
            "even when not aligned (diagonal / slightly offset panels)."
        ),
    )

    # --- title association & rendering hyperparameters ---
    ap.add_argument(
        "--margin",
        type=int,
        default=150,
        help="When collecting titles/subtitles for a cluster, expand the cluster union bbox by this margin (px).",
    )
    ap.add_argument("--pad", type=int, default=10, help="Padding (px) added around the rendered cluster canvas.")

    ap.add_argument(
        "--min_visuals_to_save",
        type=int,
        default=2,
        help="Only save a merged cluster image if at least this many visual crops were pasted.",
    )

    ap.add_argument("--font_size", type=int, default=20, help="Font size for drawing subfigure subtitles.")
    ap.add_argument("--font_path", default=None, help="Optional font path (useful for CJK). Default: auto-detect.")
    ap.add_argument(
        "--background_rgb",
        default="255,255,255",
        help="Canvas background as 'R,G,B' (default white).",
    )
    ap.add_argument("--subtitle_inset_x", type=int, default=2, help="Subtitle text X inset inside its bbox.")
    ap.add_argument("--subtitle_inset_y", type=int, default=2, help="Subtitle text Y inset inside its bbox.")
    ap.add_argument("--subtitle_bg_pad", type=int, default=2, help="Padding for subtitle background rectangle.")

    ap.add_argument(
        "--write_missing_sidecar",
        action="store_true",
        help="If set, write *.missing.txt listing visuals whose crop files were missing.",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging (prints each saved image path).",
    )

    # --- multiprocessing ---
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes (0 = auto: CPU-1). Use 1 to disable multiprocessing.",
    )
    ap.add_argument(
        "--chunksize",
        type=int,
        default=1,
        help="Pool chunksize (keep 1 for uneven paper sizes).",
    )

    return ap.parse_args()


def parse_rgb(s: str) -> Tuple[int, int, int]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Invalid --background_rgb '{s}'. Expected 'R,G,B'.")
    rgb = tuple(int(x) for x in parts)
    if any(v < 0 or v > 255 for v in rgb):
        raise ValueError(f"Invalid --background_rgb '{s}'. Values must be 0..255.")
    return rgb  # type: ignore[return-value]


def main() -> None:
    args = parse_args()

    # JSON_PATH  = "/mnt/storage/MLLM/karol/merge_sub_images/merged_subimages/all_related_text_form_image.json"
    JSON_PATH = args.base_file
    data = json.load(open(JSON_PATH))
    __cached__ = set()
    jobs = []
    for sample in data:
        if len(sample["images"])>1:
            paper_root = sample["images"][0].split("/")[:-2]
            paper_root = "/".join(paper_root)
            if paper_root not in __cached__:
                __cached__.add(paper_root)
                print(paper_root)
        else:
            paper_root = sample["images"][0][-3]
          
            import shutil
            import os 
            os.makedirs(os.path.join(args.out_root, paper_root), exist_ok=True)
            shutil.copy(sample["images"][0], os.path.join(args.out_root, paper_root))
    for paper_root in __cached__:  
        jobs.extend(find_paper_jobs(
            base_root=paper_root,
            json_subdir=args.json_subdir,
            img_subdir=args.img_subdir,
        ))             

    print(f"Found {len(jobs)} paper(s). Processing...")

    args_dict = {
        "out_root": args.out_root,
        "labels": "",
        "margin": int(args.margin),
        "pad": int(args.pad),
        "font_size": int(args.font_size),
        "font_path": args.font_path,
        "max_caption_dist": int(args.max_caption_dist),
        "no_caption_max_gap": int(args.no_caption_max_gap),
        "min_overlap": int(args.min_overlap),
        "no_caption_max_dist": int(args.no_caption_max_dist),
        "min_visuals_to_save": int(args.min_visuals_to_save),
        "background_rgb": list(parse_rgb(args.background_rgb)),
        "subtitle_inset_x": int(args.subtitle_inset_x),
        "subtitle_inset_y": int(args.subtitle_inset_y),
        "subtitle_bg_pad": int(args.subtitle_bg_pad),
        "write_missing_sidecar": bool(args.write_missing_sidecar),
        "verbose": bool(args.verbose),
    }

    if args.workers and args.workers > 0:
        workers = args.workers
    else:
        workers = max(1, (os.cpu_count() or 8) - 1)

    merged_imgs_dict: Dict[str, Dict[str, Any]] = {}
    n_errors = 0

    if workers <= 1:
        for job in tqdm(jobs, desc="papers"):
            res = _process_one_paper((job, args_dict))
            if res is None:
                continue
            paper_name, entry = res
            if isinstance(entry, dict) and "error" in entry:
                n_errors += 1
                print(f"[WARN] {paper_name} failed: {entry['error']}", file=sys.stderr)
                continue
            merged_imgs_dict[paper_name] = entry
    else:
        ctx = mp.get_context("spawn")
        payloads = ((job, args_dict) for job in jobs)

        with ctx.Pool(processes=workers) as pool:
            it = pool.imap_unordered(_process_one_paper, payloads, chunksize=max(1, args.chunksize))
            for res in tqdm(it, total=len(jobs), desc=f"papers (mp x{workers})"):
                if res is None:
                    continue
                paper_name, entry = res
                if isinstance(entry, dict) and "error" in entry:
                    n_errors += 1
                    print(f"[WARN] {paper_name} failed: {entry['error']}", file=sys.stderr)
                    continue
                merged_imgs_dict[paper_name] = entry

    dict_out = Path(args.merged_imgs_dict_out) if args.merged_imgs_dict_out else (out_root / "merged_imgs_dict.json")
    dict_out.parent.mkdir(parents=True, exist_ok=True)
    dict_out.write_text(json.dumps(merged_imgs_dict, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Number of papers with at least one merge: {len([(key, value) for key, value in merged_imgs_dict.items() if value['n_merged_imgs'] > 0])}")
    print(f"Saved merged_imgs_dict to: {dict_out.resolve()}")

    if n_errors:
        print(f"[WARN] {n_errors} paper(s) failed. See stderr output above.", file=sys.stderr)


if __name__ == "__main__":
    main()

# python merge_images_pipeline.py /mnt/storage/dataset/PPVL_reuslts_CN/中文 \
#   --max_caption_dist 450 \
#   --no_caption_max_gap 120 \
#   --min_overlap 1 \
#   --no_caption_max_dist 120 \
#   --margin 150 \
#   --out_root /mnt/workspace/MLLM/karol/merge_sub_images/merged_subimages_all_v2 \
#   --categories "现代显示,液晶与显示"
