#!/usr/bin/env python3
"""
convert_colmap_txt_to_bin.py
Converts COLMAP text format (data/buddha/*.txt) to COLMAP binary format
at data/buddha/sparse_colmap/0/{cameras.bin, images.bin, points3D.bin}
"""
import struct, os
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
TXT_DIR    = SCRIPT_DIR / "data/buddha"
OUT_DIR    = SCRIPT_DIR / "data/buddha/sparse_colmap/0"

MODEL_NUM_PARAMS = {
    'SIMPLE_PINHOLE': 3, 'PINHOLE': 4, 'SIMPLE_RADIAL': 4,
    'RADIAL': 5, 'OPENCV': 8, 'FULL_OPENCV': 12,
}
MODEL_IDS = {
    'SIMPLE_PINHOLE':0,'PINHOLE':1,'SIMPLE_RADIAL':2,'RADIAL':3,
    'OPENCV':4,'OPENCV_FISHEYE':5,'FULL_OPENCV':6,'FOV':7,
    'SIMPLE_RADIAL_FISHEYE':8,'RADIAL_FISHEYE':9,'THIN_PRISM_FISHEYE':10,
}

# ── CAMERAS ──────────────────────────────────────────────────
def convert_cameras(txt_path, out_path):
    cameras = {}
    with open(txt_path) as f:
        for line in f:
            if not line.strip() or line.startswith('#'): continue
            p = line.split()
            cid, model = int(p[0]), p[1]
            w, h = int(p[2]), int(p[3])
            params = [float(x) for x in p[4:]]
            cameras[cid] = (model, w, h, params)

    with open(out_path, 'wb') as f:
        f.write(struct.pack('<Q', len(cameras)))
        for cid, (model, w, h, params) in sorted(cameras.items()):
            mid = MODEL_IDS.get(model, 2)
            f.write(struct.pack('<iiqq', cid, mid, w, h))
            f.write(struct.pack(f'<{len(params)}d', *params))

    print(f"  ✓ cameras.bin  ({len(cameras)} cameras)")
    return cameras

# ── IMAGES ───────────────────────────────────────────────────
def convert_images(txt_path, out_path):
    images = {}  # image_id → (qvec, tvec, camera_id, name, points2d)
    with open(txt_path) as f:
        lines = [l for l in f if l.strip() and not l.startswith('#')]

    i = 0
    while i < len(lines):
        p = lines[i].split(); i += 1
        image_id  = int(p[0])
        qw,qx,qy,qz = float(p[1]),float(p[2]),float(p[3]),float(p[4])
        tx,ty,tz  = float(p[5]),float(p[6]),float(p[7])
        camera_id = int(p[8])
        name      = p[9]
        # Parse 2D points line
        pts2d = []
        if i < len(lines):
            vals = lines[i].split(); i += 1
            j = 0
            while j+2 < len(vals):
                x,y = float(vals[j]),float(vals[j+1])
                pid = int(vals[j+2]); j += 3
                pts2d.append((x, y, pid))
        images[image_id] = ([qw,qx,qy,qz],[tx,ty,tz],camera_id,name,pts2d)

    with open(out_path, 'wb') as f:
        f.write(struct.pack('<Q', len(images)))
        for iid in sorted(images):
            qvec, tvec, cid, name, pts2d = images[iid]
            f.write(struct.pack('<i', iid))
            f.write(struct.pack('<4d', *qvec))
            f.write(struct.pack('<3d', *tvec))
            f.write(struct.pack('<i', cid))
            f.write(name.encode() + b'\x00')
            f.write(struct.pack('<Q', len(pts2d)))
            for x, y, pid in pts2d:
                f.write(struct.pack('<2d', x, y))
                f.write(struct.pack('<q', pid))

    print(f"  ✓ images.bin   ({len(images)} images)")
    return images

# ── POINTS3D ──────────────────────────────────────────────────
def convert_points3d(txt_path, out_path):
    points = {}
    with open(txt_path) as f:
        for line in f:
            if not line.strip() or line.startswith('#'): continue
            p = line.split()
            pid = int(p[0])
            x,y,z = float(p[1]),float(p[2]),float(p[3])
            r,g,b = int(p[4]),int(p[5]),int(p[6])
            err   = float(p[7])
            track = []
            j = 8
            while j+1 < len(p):
                track.append((int(p[j]), int(p[j+1]))); j += 2
            points[pid] = (x,y,z,r,g,b,err,track)

    with open(out_path, 'wb') as f:
        f.write(struct.pack('<Q', len(points)))
        for pid in sorted(points):
            x,y,z,r,g,b,err,track = points[pid]
            f.write(struct.pack('<Q', pid))
            f.write(struct.pack('<3d', x, y, z))
            f.write(struct.pack('<3B', r, g, b))
            f.write(struct.pack('<d', err))
            f.write(struct.pack('<Q', len(track)))
            for img_id, pt2d_idx in track:
                f.write(struct.pack('<ii', img_id, pt2d_idx))

    print(f"  ✓ points3D.bin ({len(points)} points)")

# ── MAIN ──────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nConverting COLMAP text → binary")
    print(f"  Source : {TXT_DIR}")
    print(f"  Output : {OUT_DIR}\n")

    convert_cameras(TXT_DIR / "cameras.txt", OUT_DIR / "cameras.bin")
    convert_images( TXT_DIR / "images.txt",  OUT_DIR / "images.bin")
    convert_points3d(TXT_DIR / "points3D.txt", OUT_DIR / "points3D.bin")

    print(f"\n✓ Done — {OUT_DIR}")

if __name__ == "__main__":
    main()
