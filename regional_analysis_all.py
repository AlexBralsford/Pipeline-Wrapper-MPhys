#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch regional analysis: warps each subject’s atlas labels into diffusion (FA) space
and extracts regional FA/MD values into a CSV.

Usage:
  python3 regional_analysis_all.py \
    --processed_dir "/path/to/IN-VIVO-PROCESSED" \
    --labels_dir    "/path/to/Extracted_Files" \
    --output_csv    "/path/to/regional_metrics.csv"
"""

import os
import glob
import csv
import time
import ants
import numpy as np

def warp_labels_to_fa(fa_img, t2_img, label_path, out_path):
    """
    1) Register T2 → FA (rigid, affine, SyN)
    2) Apply transform to the label image
    """
    # 1) build registration
    reg = ants.registration(fixed=fa_img, moving=t2_img,
                        type_of_transform='SyN')  # ✅ Fixed

    # 2) warp labels into FA space (nearest neighbor interp)
    warped = ants.apply_transforms(fixed=fa_img,
                                   moving=ants.image_read(label_path),
                                   transformlist=reg['fwdtransforms'],
                                   interpolator='nearestNeighbor')
    ants.image_write(warped, out_path)

def extract_region_stats(warped_lbl_img, fa_img, md_img):
    """
    Given a label image and FA/MD images, compute per-label mean FA/MD.
    Returns dict: { label_value: (mean_fa, mean_md) }
    """
    lbl_data = warped_lbl_img.numpy().astype(int)
    fa_data = fa_img.numpy()
    md_data = md_img.numpy()
    regions = np.unique(lbl_data)
    stats = {}
    for r in regions:
        if r == 0:
            continue
        mask = lbl_data == r
        stats[r] = (
            float(np.mean(fa_data[mask])),
            float(np.mean(md_data[mask]))
        )
    return stats

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", required=True,
                        help="IN‑VIVO‑PROCESSED folder containing *_loaded subfolders")
    parser.add_argument("--labels_dir", required=True,
                        help="Extracted_Files folder, subfolders named by code (e.g. 230071)")
    parser.add_argument("--output_csv", required=True,
                        help="Where to save regional_metrics.csv")
    args = parser.parse_args()

    subjects = sorted(glob.glob(os.path.join(args.processed_dir, "*_loaded")))
    # Open CSV for writing
    with open(args.output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Header row
        writer.writerow(["subject","code","region","mean_FA","mean_MD"])
        
        for subj_path in subjects:
            subj = os.path.basename(subj_path)
            code = subj.split("_")[2]  # e.g. '230071'
            print(f"\n→  Processing {subj} (code {code})")
            
            # 1) find FA, MD
            fa_file = os.path.join(subj_path, "fa_bias_eddy.nii.gz")
            md_file = os.path.join(subj_path, "md_bias_eddy.nii.gz")
            if not os.path.exists(fa_file) or not os.path.exists(md_file):
                print(f"  ⚠️  Missing FA or MD for {subj}, skipping.")
                continue
            fa_img = ants.image_read(fa_file)
            md_img = ants.image_read(md_file)
            
            # 2) find T2 (all are named raw_T2*.nii*)
            t2_candidates = glob.glob(os.path.join(subj_path, "raw_T2*.nii*"))
            if not t2_candidates:
                print(f"  ⚠️  No raw_T2 file in {subj_path}, skipping.")
                continue
            t2_img = ants.image_read(t2_candidates[0])
            
            # 3) find label atlas for this code
            lbl_folder = os.path.join(args.labels_dir, code)
            lbl_files = glob.glob(os.path.join(lbl_folder, "*_warped_label.nii"))
            if len(lbl_files) != 1:
                print(f"  ⚠️  Found {len(lbl_files)} label files in {lbl_folder} – skipping.")
                continue
            lbl_in = lbl_files[0]
            
            # 4) warp labels into FA space
            warped_lbl = os.path.join(subj_path, f"{code}_label_in_FA.nii.gz")
            warp_labels_to_fa(fa_img, t2_img, lbl_in, warped_lbl)
            print(f"  → saved warped label: {warped_lbl}")
            
            # 5) extract per-region stats
            warped_lbl_img = ants.image_read(warped_lbl)
            stats = extract_region_stats(warped_lbl_img, fa_img, md_img)
            for region, (mfa, mmd) in stats.items():
                writer.writerow([subj, code, region, mfa, mmd])
            
            # small pause to avoid overloading
            time.sleep(1)

    print(f"\n✅  All done! Metrics saved to {args.output_csv}")

if __name__ == "__main__":
    main()
