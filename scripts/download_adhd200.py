"""
download_adhd200.py

Downloads fALFF, ReHo, and GM probability maps from the ADHD-200 CPAC pipeline
on the FCP-INDI S3 bucket, along with merged phenotypic labels.

Usage:
    python download_adhd200.py -o <out_dir> [-t <site>] [-x <sex>]
                               [--adhd-only] [--tdc-only]

Output layout:
    <out_dir>/
        phenotypic.csv              merged labels for all downloaded subjects
        <subject_id>/
            falff.nii.gz
            reho.nii.gz
            gm.nii.gz
"""

import argparse
import io
import os
import urllib.request

import pandas as pd

S3_BASE = "https://s3.amazonaws.com/fcp-indi/data/Projects/ADHD200"
CPAC_BASE = f"{S3_BASE}/Outputs/cpac/raw_outputs/pipeline_adhd200-benchmark"

# Per-site phenotypic CSV URLs (raw ADHD-200 release)
PHENO_SITE_URLS = {
    "NYU":        f"{S3_BASE}/RawData/NYU_phenotypic.csv",
    "KKI":        f"{S3_BASE}/RawData/KKI_phenotypic.csv",
    "Peking_1":   f"{S3_BASE}/RawData/Peking_1_phenotypic.csv",
    "OHSU":       f"{S3_BASE}/RawData/OHSU_phenotypic.csv",
    "NeuroIMAGE": f"{S3_BASE}/RawData/NeuroIMAGE_phenotypic.csv",
    "Pittsburgh": f"{S3_BASE}/RawData/Pittsburgh_phenotypic.csv",
}

# CPAC derivative sub-paths relative to <subject>_session_1/
# Prefer global1 (global signal regression); fall back to global0 if absent.
_SELECTOR_G1 = (
    "/_compcor_ncomponents_5_selector_pc10.linear1.wm0.global1"
    ".motion1.quadratic1.gm0.compcor1.csf0"
)
_SELECTOR_G0 = (
    "/_compcor_ncomponents_5_selector_pc10.linear1.wm0.global0"
    ".motion1.quadratic1.gm0.compcor1.csf0"
)
_FALFF_STEM = (
    "falff_to_standard/_scan_rest_1"
    "/_csf_threshold_0.96/_gm_threshold_0.7/_wm_threshold_0.96"
)
_REHO_STEM = (
    "reho_to_standard/_scan_rest_1"
    "/_csf_threshold_0.96/_gm_threshold_0.7/_wm_threshold_0.96"
)

FALFF_SUBPATHS = [
    f"{_FALFF_STEM}{_SELECTOR_G1}/_hp_0.01/_lp_0.1/rest_calc_tshift_resample_volreg_mask_calc_antswarp.nii.gz",
    f"{_FALFF_STEM}{_SELECTOR_G0}/_hp_0.01/_lp_0.1/rest_calc_tshift_resample_volreg_mask_calc_antswarp.nii.gz",
]
REHO_SUBPATHS = [
    f"{_REHO_STEM}{_SELECTOR_G1}/ReHo_antswarp.nii.gz",
    f"{_REHO_STEM}{_SELECTOR_G0}/ReHo_antswarp.nii.gz",
]
GM_SUBPATH = "seg_probability_maps/segment_prob_1.nii.gz"

DERIVATIVES = {
    "falff": FALFF_SUBPATHS,
    "reho":  REHO_SUBPATHS,
    "gm":    [GM_SUBPATH],
}

# DX column: 0=TDC, 1=ADHD-Combined, 2=ADHD-Hyperactive, 3=ADHD-Inattentive
TDC_DX  = 0
ADHD_DX = {1, 2, 3}


def load_phenotypic(site_filter=None, sex_filter=None,
                    adhd_only=False, tdc_only=False):
    """Download and merge per-site phenotypic CSVs. Returns a DataFrame."""
    frames = []
    for site, url in PHENO_SITE_URLS.items():
        if site_filter and site.lower() != site_filter.lower():
            continue
        try:
            raw = urllib.request.urlopen(url).read().decode()
            df = pd.read_csv(io.StringIO(raw))
            df["SITE"] = site
            frames.append(df)
        except Exception as e:
            print(f"Warning: could not load phenotypic for {site}: {e}")

    if not frames:
        raise RuntimeError("No phenotypic data loaded.")

    pheno = pd.concat(frames, ignore_index=True)
    pheno = pheno.rename(columns={
        "ScanDir ID": "subject_id",
        "Gender":     "sex",       # 0=female, 1=male
        "Age":        "age",
        "DX":         "dx",
        "QC_Rest_1":  "qc_rest",
    })
    pheno["subject_id"] = pheno["subject_id"].astype(str).str.zfill(7)
    pheno["dx"]       = pd.to_numeric(pheno["dx"],       errors="coerce")
    pheno["qc_rest"]  = pd.to_numeric(pheno["qc_rest"],  errors="coerce")

    # Quality filter: keep only subjects with good resting-state QC
    pheno = pheno[pheno["qc_rest"] == 1]

    # Diagnosis filter
    if adhd_only:
        pheno = pheno[pheno["dx"].isin(ADHD_DX)]
    elif tdc_only:
        pheno = pheno[pheno["dx"] == TDC_DX]
    else:
        pheno = pheno[pheno["dx"].isin(ADHD_DX | {TDC_DX})]

    # Sex filter: 'M'=male(1), 'F'=female(0)
    if sex_filter == "M":
        pheno = pheno[pheno["sex"] == 1]
    elif sex_filter == "F":
        pheno = pheno[pheno["sex"] == 0]

    # Binary label: 0=ADHD, 1=TDC
    pheno["label"] = (pheno["dx"] == TDC_DX).astype(int)

    return pheno.reset_index(drop=True)


def list_cpac_subjects():
    """Return the set of subject IDs present in the CPAC S3 pipeline."""
    url = (
        "https://s3.amazonaws.com/fcp-indi"
        "?prefix=data/Projects/ADHD200/Outputs/cpac/raw_outputs/"
        "pipeline_adhd200-benchmark/&delimiter=/"
    )
    xml = urllib.request.urlopen(url).read().decode()
    subjects = set()
    for chunk in xml.split("<Prefix>"):
        if "_session_1/" in chunk:
            sid = chunk.split("benchmark/")[1].split("_session_1/")[0]
            subjects.add(sid.zfill(7))
    return subjects


def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"  exists, skipping: {dest_path}")
        return True
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest_path)
        return True
    except Exception as e:
        print(f"  FAILED {url}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def collect_and_download(out_dir, site_filter=None, sex_filter=None,
                         adhd_only=False, tdc_only=False):
    os.makedirs(out_dir, exist_ok=True)

    print("Loading phenotypic data...")
    pheno = load_phenotypic(site_filter, sex_filter, adhd_only, tdc_only)
    print(f"  {len(pheno)} subjects pass phenotypic filters "
          f"({(pheno['label']==0).sum()} ADHD, {(pheno['label']==1).sum()} TDC)")

    print("Listing CPAC subjects on S3...")
    cpac_subjects = list_cpac_subjects()
    print(f"  {len(cpac_subjects)} subjects in CPAC pipeline")

    pheno_ids = set(pheno["subject_id"].tolist())
    matched = sorted(pheno_ids & cpac_subjects)
    print(f"  {len(matched)} subjects matched to phenotypic data")

    # Save merged phenotypic CSV for matched subjects only
    pheno_matched = pheno[pheno["subject_id"].isin(matched)].copy()
    pheno_matched.to_csv(os.path.join(out_dir, "phenotypic.csv"), index=False)

    # Download derivatives
    total = len(matched) * len(DERIVATIVES)
    done = 0
    failed = []

    for subject_id in matched:
        subject_dir = f"{subject_id}_session_1"
        for deriv_name, subpaths in DERIVATIVES.items():
            dest = os.path.join(out_dir, subject_id, f"{deriv_name}.nii.gz")
            done += 1
            print(f"[{done}/{total}] {subject_id} / {deriv_name}")
            success = False
            for subpath in subpaths:
                url = f"{CPAC_BASE}/{subject_dir}/{subpath}"
                if download_file(url, dest):
                    success = True
                    break
            if not success:
                failed.append((subject_id, deriv_name))

    print(f"\nDone. {done - len(failed)}/{total} files downloaded.")
    if failed:
        print(f"Failed ({len(failed)}): {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--out_dir", required=True,
                        help="Local directory to save files")
    parser.add_argument("-t", "--site", default=None,
                        help="Site filter, e.g. NYU, KKI, Peking_1")
    parser.add_argument("-x", "--sex", default=None,
                        help="Sex filter: M or F")
    parser.add_argument("--adhd-only", action="store_true")
    parser.add_argument("--tdc-only", action="store_true")
    args = parser.parse_args()

    collect_and_download(
        out_dir=os.path.abspath(args.out_dir),
        site_filter=args.site,
        sex_filter=args.sex,
        adhd_only=args.adhd_only,
        tdc_only=args.tdc_only,
    )
