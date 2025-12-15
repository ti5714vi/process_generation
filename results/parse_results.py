
# Exectute with "python ./parse_results.py <PATH_TO_DIR_WITH_FILES>"

import os
import re
import argparse
import subprocess
import tempfile
import gzip
import math
from collections import defaultdict

# ---------------------------
# Regexes to parse log files
# ---------------------------
POINTS_RE = re.compile(r"(\d+)\s+points")
TIME_RE = re.compile(r"Total time:\s*([0-9]*\.?[0-9]+)")
XJ_RE = re.compile(r"([0-9])j_")

# Matches lines like:
# Integral (accum): 0.137194E+09 +/- 0.1184E+06 ( 0.0863 %)
INTEGRAL_RE = re.compile(
    r"Integral\s*\(accum\):\s*"
    r"([0-9]*\.?[0-9]+(?:[Ee][+\-]?\d+)?)\s*"
    r"\+/\-\s*([0-9]*\.?[0-9]+(?:[Ee][+\-]?\d+)?)\s*"
    r"\(\s*([0-9]*\.?[0-9]+)\s*%\s*\)"
)

# ---------------------------
# Helpers for unwgt & LHE log parsing
# ---------------------------
# Robust float detection: handles plain, decimal, and scientific-notation numbers
_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+\.|\.\d+|\d+)(?:[eE][+-]?\d+)?")

def _first_float_in_line(line: str):
    """Return the first float found in a text line, or None if no float present."""
    m = _FLOAT_RE.search(line)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


# ---------------------------
# Filename parsing helpers
# ---------------------------
def normalize_group_label(filename):
    idx = filename.find("s1")
    if idx == -1:
        return None
    return filename[:idx + 2]  # include 's1'


def extract_x_and_base_prefix(filename):
    m = XJ_RE.search(filename)
    if not m:
        return None, None
    x_val = int(m.group(1))
    base_prefix = filename[:m.start()]
    return x_val, base_prefix


# ---------------------------
# Per-file extractors
# ---------------------------
def sum_points_in_file(filepath):
    total = 0
    with open(filepath, "r") as f:
        for line in f:
            for match in POINTS_RE.findall(line):
                total += int(match)
    return total


def extract_time_in_file(filepath):
    with open(filepath, "r") as f:
        for line in f:
            m = TIME_RE.search(line)
            if m:
                return float(m.group(1))
    return None


def extract_final_integral(filepath):
    """
    Return (xsec, unc, rel_percent) from the FINAL 'Integral (accum)' line.
    If none found, return None.
    """
    last = None
    with open(filepath, "r") as f:
        for line in f:
            m = INTEGRAL_RE.search(line)
            if m:
                xsec = float(m.group(1))
                unc = float(m.group(2))
                rel = float(m.group(3))
                last = (xsec, unc, rel)
    return last  # Could be None if no match


# ---------------------------
# Cached overweight fraction from LHE gz + check_wgts
# ---------------------------
def _check_wgts_cache_path(lhe_gz_path: str) -> str:
    """
    Cache file path for the stdout of `check_wgts` that was run on the given events.lhe.gz.
    The cache lives next to the .gz file with a clear suffix.
    """
    return lhe_gz_path + ".check_wgts.out"


def extract_overweight_fraction(txt_filename, directory):
    """
    Return the 'overweight fraction' for a given log by using its paired events.lhe.gz.

    Caching behavior:
    - First run: unzip to a temp .lhe, run `../master_clean/Utilities/check_wgts`,
      write the full stdout to a cache file, parse the last line's last token as float.
    - Subsequent runs: if a cache file exists and is newer than (or same age as) the .gz,
      read stdout from the cache and parse the value—no unzip/exec needed.
    """
    lhe_gz_filename = txt_filename.replace("log_file.txt", "events.lhe.gz")
    lhe_gz_path = os.path.join(directory, lhe_gz_filename)

    if not os.path.isfile(lhe_gz_path):
        print(f"Warning: no corresponding LHE file for {txt_filename}")
        return None

    cache_path = _check_wgts_cache_path(lhe_gz_path)

    # Decide whether to use cache (cache_mtime >= gz_mtime)
    try:
        gz_mtime = os.path.getmtime(lhe_gz_path)
        cache_exists = os.path.exists(cache_path)
        cache_mtime = os.path.getmtime(cache_path) if cache_exists else -1
    except Exception:
        gz_mtime = -1
        cache_exists = False
        cache_mtime = -1

    if cache_exists and cache_mtime >= gz_mtime:
        try:
            with open(cache_path, "r") as cf:
                lines = cf.read().strip().splitlines()
            if not lines:
                return None
            last_line = lines[-1]
            overweight = float(last_line.split()[-1])
            return overweight
        except Exception as e:
            print(f"Warning: failed to read/parse cached check_wgts output '{cache_path}': {e}")
            # fall through to recompute below

    # Cache missing or stale: recompute via unzip + external tool
    tmp_lhe_path = None
    try:
        with gzip.open(lhe_gz_path, "rb") as gzfile:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".lhe") as tmpfile:
                tmpfile.write(gzfile.read())
                tmp_lhe_path = tmpfile.name

        result = subprocess.run(
            ["../master_clean/Utilities/check_wgts", tmp_lhe_path],
            capture_output=True, text=True, check=True
        )
        stdout_text = result.stdout

        # Write stdout to cache atomically
        try:
            tmp_dir = os.path.dirname(cache_path) or "."
            with tempfile.NamedTemporaryFile(delete=False, dir=tmp_dir, suffix=".tmp") as tmpcache:
                tmpcache.write(stdout_text.encode("utf-8"))
                tmpcache_path = tmpcache.name
            os.replace(tmpcache_path, cache_path)  # atomic on POSIX
        except Exception as e:
            print(f"Warning: failed to write cache '{cache_path}': {e}")

        lines = stdout_text.strip().splitlines()
        overweight = float(lines[-1].split()[-1]) if lines else None
        return overweight
    except Exception as e:
        print(f"Warning: failed to process {lhe_gz_filename}: {e}")
        return None
    finally:
        # Clean up the temporary .lhe if it was created
        try:
            if tmp_lhe_path and os.path.isfile(tmp_lhe_path):
                os.remove(tmp_lhe_path)
        except Exception:
            pass


# ---------------------------
# Extract UnwEff & ESS from out_unwgt_.txt
# ---------------------------
def extract_unw_eff_and_ess(txt_filename: str, directory: str):
    """
    For a given '*log_file.txt', read the corresponding '*out_unwgt_.txt' file.

    We return:
      - unw_eff: first float on the FIRST line
      - ess:     first float on the LAST  line

    If file is missing or parsing fails, return (None, None).
    """
    unw_filename = txt_filename.replace("log_file.txt", "out_unwgt_.txt")
    unw_path = os.path.join(directory, unw_filename)
    if not os.path.isfile(unw_path):
        return None, None

    try:
        with open(unw_path, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines:
            return None, None

        # First line → first number (unweighting efficiency)
        unw_eff = _first_float_in_line(lines[0])

        # Last line → first number (ESS)
        ess = _first_float_in_line(lines[-1])

        return unw_eff, ess
    except Exception:
        return None, None


# ---------------------------
# Extract LHE log metrics from events.lhe.log
# ---------------------------
def extract_lhe_log_metrics(txt_filename: str, directory: str):
    """
    For a given '*log_file.txt', read the corresponding 'events.lhe.log' file.

    We parse:
      - lhe_log_time: float from line starting 'Total time:'
      - fc_xsec:      float from line starting 'Total FC cross section:'

    If file is missing or parsing fails, return (None, None).
    """
    lhe_log_filename = txt_filename.replace("log_file.txt", "events.lhe.log")
    lhe_log_path = os.path.join(directory, lhe_log_filename)
    if not os.path.isfile(lhe_log_path):
        return None, None

    lhe_log_time = None
    fc_xsec = None
    try:
        with open(lhe_log_path, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith("Total time:"):
                    # first float after label
                    val = _first_float_in_line(line)
                    if val is not None:
                        lhe_log_time = val
                elif line.startswith("Total FC cross section:"):
                    val = _first_float_in_line(line)
                    if val is not None:
                        fc_xsec = val

        return lhe_log_time, fc_xsec
    except Exception:
        return None, None


# ---------------------------
# Compatibility stats
# ---------------------------
def compatibility_stats(xsecs, uncs):
    """
    Compute weighted mean, its uncertainty, chi2, dof, reduced chi2, p-value (if SciPy available),
    standardized residuals, Birge ratio, and max pairwise z between seeds.
    """
    assert len(xsecs) == len(uncs) and len(xsecs) > 0
    n = len(xsecs)

    # Weights
    w = [1.0 / (u * u) if u > 0.0 else 0.0 for u in uncs]
    W = sum(w)

    # Guard against zero-uncertainty or all-zero weights
    if W == 0.0:
        # Fall back to simple mean with naive uncertainty
        xhat = sum(xsecs) / n
        sigma_xhat = float('nan')
        chi2 = float('nan')
        dof = n - 1
        red_chi2 = float('nan')
        birge = float('nan')
        p_value = None
        residuals = [float('nan')] * n
        max_pairwise_z = float('nan')
        return {
            "weighted_mean": xhat,
            "weighted_unc": sigma_xhat,
            "chi2": chi2,
            "dof": dof,
            "red_chi2": red_chi2,
            "p_value": p_value,
            "birge_ratio": birge,
            "std_residuals": residuals,
            "max_pairwise_z": max_pairwise_z,
        }

    xhat = sum(wi * xi for wi, xi in zip(w, xsecs)) / W
    sigma_xhat = math.sqrt(1.0 / W)

    # Chi-square against weighted mean
    chi2 = sum(((xi - xhat) ** 2) / (ui * ui) for xi, ui in zip(xsecs, uncs) if ui > 0.0)
    dof = n - 1
    red_chi2 = chi2 / dof if dof > 0 else float('nan')
    birge = math.sqrt(red_chi2) if dof > 0 else float('nan')

    # p-value (if SciPy is available)
    try:
        from scipy.stats import chi2 as chi2_dist
        p_value = 1.0 - chi2_dist.cdf(chi2, df=dof) if dof > 0 else float('nan')
    except Exception:
        p_value = None  # SciPy not available

    # Standardized residuals
    residuals = [(xi - xhat) / ui if ui > 0.0 else float('inf') for xi, ui in zip(xsecs, uncs)]

    # Max pairwise z-score
    max_pairwise_z = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            denom = math.sqrt(uncs[i] ** 2 + uncs[j] ** 2)
            if denom > 0:
                zij = abs(xsecs[i] - xsecs[j]) / denom
                if zij > max_pairwise_z:
                    max_pairwise_z = zij
    if n < 2:
        max_pairwise_z = float('nan')

    return {
        "weighted_mean": xhat,
        "weighted_unc": sigma_xhat,
        "chi2": chi2,
        "dof": dof,
        "red_chi2": red_chi2,
        "p_value": p_value,  # None if SciPy missing
        "birge_ratio": birge,
        "std_residuals": residuals,
        "max_pairwise_z": max_pairwise_z,
    }


def compatibility_stats_for_group(xsecs, uncs, alpha=0.05, redchi2_thresh=2.0, apply_birge=False):
    """
    Wrap compatibility_stats, decide 'compatible' flag and optionally Birge-scale the combined uncertainty.
    Returns a dict with formatted values ready for printing.
    """
    stats = compatibility_stats(xsecs, uncs)

    # Decide compatibility
    if stats["dof"] <= 0:
        compatible = "N/A"
    elif stats["p_value"] is not None:
        compatible = "Yes" if stats["p_value"] >= alpha else "No"
    else:
        compatible = "Yes" if stats["red_chi2"] <= redchi2_thresh else "No"

    # Optionally Birge-scale the combined uncertainty if excess scatter
    combined_unc = stats["weighted_unc"]
    if apply_birge and stats["dof"] > 0 and stats["red_chi2"] > 1.0 and math.isfinite(stats["birge_ratio"]):
        combined_unc = combined_unc * stats["birge_ratio"]

    # Format numbers
    fmt3e = lambda v: ("N/A" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.3e}")
    fmt_p = lambda p: ("NA" if p is None or (isinstance(p, float) and not math.isfinite(p)) else f"{p:.2e}")
    chi_str = (f"{stats['chi2']:.2f}/{stats['dof']}" if stats["dof"] > 0 and math.isfinite(stats["chi2"]) else "N/A")

    return {
        "weighted_mean": fmt3e(stats["weighted_mean"]),
        "weighted_unc": fmt3e(combined_unc),
        "chi2_over_dof": chi_str,
        "red_chi2": fmt3e(stats["red_chi2"]),
        "p_value": fmt_p(stats["p_value"]),
        "birge": fmt3e(stats["birge_ratio"]),
        "max_zij": fmt3e(stats["max_pairwise_z"]),
        "compatible": compatible,
    }


# ---------------------------
# Directory processing
# ---------------------------
def process_directory(directory):
    groups_points = defaultdict(list)
    groups_time = defaultdict(list)
    groups_overweight = defaultdict(list)
    groups_xsec = defaultdict(list)
    groups_xsec_unc = defaultdict(list)
    # NEW: unweighting efficiency and ESS
    groups_unweff = defaultdict(list)
    groups_ess = defaultdict(list)
    # NEW: metrics from events.lhe.log
    groups_lhe_time = defaultdict(list)
    groups_fc_xsec = defaultdict(list)
    groups_fc_rwgt = defaultdict(list)

    # Optional if you ever need it:
    # groups_xsec_rel = defaultdict(list)
    base_prefix_map = {}

    for filename in os.listdir(directory):
        if not (filename.endswith("log_file.txt") and "s1" in filename):
            continue

        group_label = normalize_group_label(filename)
        x_val, base_prefix = extract_x_and_base_prefix(filename)
        if group_label is None or x_val is None or base_prefix is None:
            continue

        filepath = os.path.join(directory, filename)
        if not os.path.isfile(filepath):
            continue

        # Points
        points = sum_points_in_file(filepath)
        groups_points[(group_label, x_val)].append(points)

        # Time (from log_file.txt)
        time_val = extract_time_in_file(filepath)
        if time_val is not None:
            groups_time[(group_label, x_val)].append(time_val)
        else:
            print(f"Warning: '{filename}' has no 'Total time:' entry, skipping for time stats.")

        # Cross section (final 'Integral (accum)' line only)
        integral = extract_final_integral(filepath)
        if integral is not None:
            xsec, unc, rel = integral
            groups_xsec[(group_label, x_val)].append(xsec)
            groups_xsec_unc[(group_label, x_val)].append(unc)
            # groups_xsec_rel[(group_label, x_val)].append(rel)
        else:
            print(f"Warning: '{filename}' has no final 'Integral (accum)' entry, skipping for xsec stats.")

        # Overweight fraction from LHE gz (cached)
        overweight_val = extract_overweight_fraction(filename, directory)
        if overweight_val is not None:
            groups_overweight[(group_label, x_val)].append(overweight_val)

        # Unweighting efficiency & ESS from out_unwgt_.txt
        unw_eff, ess = extract_unw_eff_and_ess(filename, directory)
        if unw_eff is not None:
            groups_unweff[(group_label, x_val)].append(unw_eff)
        if ess is not None:
            groups_ess[(group_label, x_val)].append(ess)

        # NEW: LHE log metrics from events.lhe.log
        lhe_log_time, fc_xsec = extract_lhe_log_metrics(filename, directory)
        if lhe_log_time is not None:
            groups_lhe_time[(group_label, x_val)].append(lhe_log_time)
        if fc_xsec is not None:
            groups_fc_xsec[(group_label, x_val)].append(fc_xsec)
            groups_fc_rwgt[(group_label, x_val)].append(fc_xsec/xsec)

        # Track base_prefix for sorting and block separation
        base_prefix_map[(group_label, x_val)] = base_prefix

    return (
        groups_points, groups_time, groups_overweight,
        groups_xsec, groups_xsec_unc,
        groups_unweff, groups_ess,
        groups_lhe_time, groups_fc_xsec, groups_fc_rwgt,
        base_prefix_map
    )


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute per-group stats for points, times, overweight fraction, cross section, UnwEff/ESS, and LHE log metrics, with compatibility checks"
    )
    parser.add_argument("directory", help="Path to the directory containing files")
    parser.add_argument("--compat-alpha", type=float, default=0.05,
                        help="Significance level for p-value compatibility (default: 0.05)")
    parser.add_argument("--compat-redchi2-thresh", type=float, default=2.0,
                        help="Fallback red chi^2 threshold when p-value is unavailable (default: 2.0)")
    parser.add_argument("--apply-birge", action="store_true",
                        help="If set, Birge-scale the combined uncertainty when there is excess scatter")
    args = parser.parse_args()

    (groups_points, groups_time, groups_overweight,
     groups_xsec, groups_xsec_unc,
     groups_unweff, groups_ess,
     groups_lhe_time, groups_fc_xsec, groups_fc_rwgt,
     base_prefix_map) = process_directory(args.directory)

    # Prepare compatibility per key (union of keys where both xsec and unc exist)
    groups_compat = {}
    for key in set(groups_xsec.keys()) | set(groups_xsec_unc.keys()):
        xsecs = groups_xsec.get(key, [])
        uncs = groups_xsec_unc.get(key, [])
        if len(xsecs) > 0 and len(uncs) == len(xsecs):
            groups_compat[key] = compatibility_stats_for_group(
                xsecs, uncs,
                alpha=args.compat_alpha,
                redchi2_thresh=args.compat_redchi2_thresh,
                apply_birge=args.apply_birge
            )

    # Table header
    print(f"{'Group':<24} {'X':<5} "
          f"{'Points(avg)':>15} {'Points(min)':>15} {'Points(max)':>15} "
          f"{'Time(avg)':>15} {'Time(min)':>15} {'Time(max)':>15} "
          f"{'Overweight(avg)':>18} {'Overweight(min)':>18} {'Overweight(max)':>18} "
          f"{'Xsec(avg)':>15} {'Xsec(min)':>15} {'Xsec(max)':>15} "
          f"{'Unc(avg)':>15} {'Unc(min)':>15} {'Unc(max)':>15} "
          # Weighted mean & compatibility
          f"{'Xsec(w-mean)':>15} {'Unc(w-mean)':>15} "
          f"{'chi2/dof':>12} {'redChi2':>12} {'p-value':>10} {'Birge':>12} {'max z_ij':>12} {'Compat':>10}"
          # UnwEff & ESS
          f"{'UnwEff(avg)':>15} {'UnwEff(min)':>15} {'UnwEff(max)':>15} "
          f"{'ESS(avg)':>15} {'ESS(min)':>15} {'ESS(max)':>15} "
          # NEW: LHE log metrics (Total time, FC cross section)
          f"{'rwgtTime(avg)':>17} {'rwgtTime(min)':>17} {'rwgtTime(max)':>17} "
          f"{'FCXsec(avg)':>15} {'FCXsec(min)':>15} {'FCXsec(max)':>15} "
          f"{'FCrwgt(avg)':>15} {'FCrwgt(min)':>15} {'FCrwgt(max)':>15} "
          )
    print("-" * 420)
    print()
    print()

    # Determine all keys to print (union of all groups)
    all_keys = sorted(
        set(groups_points.keys())
        | set(groups_time.keys())
        | set(groups_overweight.keys())
        | set(groups_xsec.keys())
        | set(groups_xsec_unc.keys())
        | set(groups_unweff.keys())
        | set(groups_ess.keys())
        | set(groups_lhe_time.keys())     # NEW
        | set(groups_fc_xsec.keys())      # NEW
        | set(groups_fc_rwgt.keys())      # NEW
        | set(groups_compat.keys()),
        key=lambda k: (base_prefix_map.get(k, ""), k[0], k[1])
    )

    prev_base = None
    for (group_label, x_val) in all_keys:
        # Points stats
        if (group_label, x_val) in groups_points:
            pvals = groups_points[(group_label, x_val)]
            avg_points = round(sum(pvals) / len(pvals))
            min_points = round(min(pvals))
            max_points = round(max(pvals))
        else:
            avg_points = min_points = max_points = "N/A"

        # Time stats (scientific notation, 3 sig figs)
        if (group_label, x_val) in groups_time:
            tvals = groups_time[(group_label, x_val)]
            avg_time = sum(tvals) / len(tvals)
            min_time = min(tvals)
            max_time = max(tvals)
            time_fmt = lambda v: f"{v:.3e}"
            avg_time, min_time, max_time = time_fmt(avg_time), time_fmt(min_time), time_fmt(max_time)
        else:
            avg_time = min_time = max_time = "N/A"

        # Overweight stats (scientific notation, 3 sig figs)
        if (group_label, x_val) in groups_overweight:
            ovals = groups_overweight[(group_label, x_val)]
            avg_over = sum(ovals) / len(ovals)
            min_over = min(ovals)
            max_over = max(ovals)
            over_fmt = lambda v: f"{v:.3e}"
            avg_over, min_over, max_over = over_fmt(avg_over), over_fmt(min_over), over_fmt(max_over)
        else:
            avg_over = min_over = max_over = "N/A"

        # Cross section stats (scientific notation, 3 sig figs)
        if (group_label, x_val) in groups_xsec:
            xs = groups_xsec[(group_label, x_val)]
            avg_xs = sum(xs) / len(xs)
            min_xs = min(xs)
            max_xs = max(xs)
            xs_fmt = lambda v: f"{v:.3e}"
            avg_xs, min_xs, max_xs = xs_fmt(avg_xs), xs_fmt(min_xs), xs_fmt(max_xs)
        else:
            avg_xs = min_xs = max_xs = "N/A"

        # Uncertainty stats (scientific notation, 3 sig figs)
        if (group_label, x_val) in groups_xsec_unc:
            us = groups_xsec_unc[(group_label, x_val)]
            avg_u = sum(us) / len(us)
            min_u = min(us)
            max_u = max(us)
            u_fmt = lambda v: f"{v:.3e}"
            avg_u, min_u, max_u = u_fmt(avg_u), u_fmt(min_u), u_fmt(max_u)
        else:
            avg_u = min_u = max_u = "N/A"

        # Unweighting efficiency (scientific notation)
        if (group_label, x_val) in groups_unweff:
            uev = groups_unweff[(group_label, x_val)]
            avg_unweff = sum(uev) / len(uev)
            min_unweff = min(uev)
            max_unweff = max(uev)
            fmt = lambda v: f"{v:.3e}"
            avg_unweff, min_unweff, max_unweff = fmt(avg_unweff), fmt(min_unweff), fmt(max_unweff)
        else:
            avg_unweff = min_unweff = max_unweff = "N/A"

        # ESS (scientific notation)
        if (group_label, x_val) in groups_ess:
            essv = groups_ess[(group_label, x_val)]
            avg_ess = sum(essv) / len(essv)
            min_ess = min(essv)
            max_ess = max(essv)
            fmt = lambda v: f"{v:.3e}"
            avg_ess, min_ess, max_ess = fmt(avg_ess), fmt(min_ess), fmt(max_ess)
        else:
            avg_ess = min_ess = max_ess = "N/A"

        # NEW: LHE log time (scientific notation)
        if (group_label, x_val) in groups_lhe_time:
            ltvals = groups_lhe_time[(group_label, x_val)]
            avg_lhe_t = sum(ltvals) / len(ltvals)
            min_lhe_t = min(ltvals)
            max_lhe_t = max(ltvals)
            fmt = lambda v: f"{v:.3e}"
            avg_lhe_t, min_lhe_t, max_lhe_t = fmt(avg_lhe_t), fmt(min_lhe_t), fmt(max_lhe_t)
        else:
            avg_lhe_t = min_lhe_t = max_lhe_t = "N/A"

        # NEW: Final FC cross section (scientific notation)
        if (group_label, x_val) in groups_fc_xsec:
            fcx = groups_fc_xsec[(group_label, x_val)]
            avg_fcx = sum(fcx) / len(fcx)
            min_fcx = min(fcx)
            max_fcx = max(fcx)
            fmt = lambda v: f"{v:.3e}"
            avg_fcx, min_fcx, max_fcx = fmt(avg_fcx), fmt(min_fcx), fmt(max_fcx)
        else:
            avg_fcx = min_fcx = max_fcx = "N/A"

        # NEW: Final FC cross section (scientific notation)
        if (group_label, x_val) in groups_fc_rwgt:
            fcr = groups_fc_rwgt[(group_label, x_val)]
            avg_fcr = sum(fcr) / len(fcr)
            min_fcr = min(fcr)
            max_fcr = max(fcr)
            fmt = lambda v: f"{v:.3e}"
            avg_fcr, min_fcr, max_fcr = fmt(avg_fcr), fmt(min_fcr), fmt(max_fcr)
        else:
            avg_fcr = min_fcr = max_fcr = "N/A"

        # Compatibility summary
        compat = groups_compat.get((group_label, x_val), None)
        if compat is not None:
            wmean = compat["weighted_mean"]
            wunc = compat["weighted_unc"]
            chi_over_dof = compat["chi2_over_dof"]
            redchi2 = compat["red_chi2"]
            pval = compat["p_value"]
            birge = compat["birge"]
            maxzij = compat["max_zij"]
            compat_flag = compat["compatible"]
        else:
            wmean = wunc = redchi2 = pval = birge = maxzij = "N/A"
            chi_over_dof = "N/A"
            compat_flag = "N/A"

        base_prefix = base_prefix_map.get((group_label, x_val), "")
        if prev_base is not None and base_prefix != prev_base:
            print()
            print()

        # Always print the row
        print(f"{group_label:<24} {x_val:<5} "
              f"{str(avg_points):>15} {str(min_points):>15} {str(max_points):>15} "
              f"{avg_time:>15} {min_time:>15} {max_time:>15} "
              f"{avg_over:>18} {min_over:>18} {max_over:>18} "
              f"{avg_xs:>15} {min_xs:>15} {max_xs:>15} "
              f"{avg_u:>15} {min_u:>15} {max_u:>15} "
              f"{wmean:>15} {wunc:>15} "
              f"{chi_over_dof:>12} {redchi2:>12} {pval:>10} {birge:>12} {maxzij:>12} {compat_flag:>10}"
              f"{avg_unweff:>15} {min_unweff:>15} {max_unweff:>15} "
              f"{avg_ess:>15} {min_ess:>15} {max_ess:>15} "
              f"{avg_lhe_t:>17} {min_lhe_t:>17} {max_lhe_t:>17} "
              f"{avg_fcx:>15} {min_fcx:>15} {max_fcx:>15} "
              f"{avg_fcr:>15} {min_fcr:>15} {max_fcr:>15} "
              )

        prev_base = base_prefix
