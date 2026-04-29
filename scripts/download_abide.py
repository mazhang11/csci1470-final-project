"""
download_abide.py

Downloads fALFF, ReHo, and GM probability maps from the ABIDE CPAC pipeline
on the FCP-INDI S3 bucket, along with merged phenotypic labels.
"""

import argparse
import os
import urllib.request
import pandas as pd

S3_BASE = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative"
PHENO_URL = f"{S3_BASE}/Phenotypic_V1_0b_preprocessed1.csv"
DERIVATIVES = ["falff", "reho", "gm"]

def load_phenotypic():
    print("Downloading ABIDE Phenotypic CSV...")
    df = pd.read_csv(PHENO_URL)
    
    # Standardize columns to match dataset.py expectations
    df = df.rename(columns={
        "FILE_ID": "subject_id",
        "DX_GROUP": "dx"
    })
    
    df = df.dropna(subset=["subject_id", "dx"])
    
    # DX_GROUP: 1 = Autism, 2 = Control
    # Our format: 1 = Control (TDC), 0 = Disorder (ASD)
    df["label"] = (df["dx"] == 2).astype(int)
    return df

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        return True
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest_path)
        return True
    except Exception:
        # Fails silently and cleans up empty file if subject is missing a specific scan
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False

def collect_and_download(out_dir, max_subjects=None):
    os.makedirs(out_dir, exist_ok=True)
    pheno = load_phenotypic()
    
    if max_subjects:
        pheno = pheno.head(max_subjects)
        
    matched = pheno["subject_id"].tolist()
    
    # Save labels
    pheno.to_csv(os.path.join(out_dir, "phenotypic.csv"), index=False)
    
    total = len(matched) * len(DERIVATIVES)
    done = 0
    
    for subject_id in matched:
        for deriv in DERIVATIVES:
            dest = os.path.join(out_dir, subject_id, f"{deriv}.nii.gz")
            done += 1
            print(f"[{done}/{total}] ABIDE {subject_id} / {deriv}")
            
            # ABIDE CPAC finalized paths
            if deriv == "falff":
                url = f"{S3_BASE}/Outputs/cpac/filt_global/falff/{subject_id}_falff.nii.gz"
            elif deriv == "reho":
                url = f"{S3_BASE}/Outputs/cpac/filt_global/reho/{subject_id}_reho.nii.gz"
            elif deriv == "gm":
                url = f"{S3_BASE}/Outputs/cpac/rois_gm/{subject_id}_rois_gm.nii.gz"
                
            download_file(url, dest)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out_dir", required=True)
    parser.add_argument("--max", type=int, default=None, help="Limit downloads for testing")
    args = parser.parse_args()
    collect_and_download(os.path.abspath(args.out_dir), args.max)