#!/usr/bin/env python3
"""Top-level SNOWWI SAR processing driver.

Runs a full flightline through the MIRSL SNOWWI chain locally, materialising
per-region config as it goes:

    setup -> novatel -> peg -> preprocess -> [output-map] -> azmcomp
          -> postprocess -> [estimate_height] -> [geolocate]

Two modes, selected by whether a FLIGHTLINE is given:

* single flightline:  ``process_snowwi.py 20260129 GrandMesa2 -b low``
* whole date:         ``process_snowwi.py 20260129 -b low --flightline-db DB``
  (iterates every flightline flown on that date, from the flightline DB).

Region (pulse length, receive delay, DEM) is inferred from the flightline name.
Waveform parameters come from ``snowwi_tools.params.SIGNAL_PARAMS`` -- the
single source of truth; nothing is re-hardcoded here. Peg comes from a
``pegs_<campaign>.txt`` file (output of ``get_peg_points.py``).

Author: generated for the MIRSL SNOWWI processor.
"""

import argparse
import glob
import os
import re
import subprocess
import sys
from pathlib import Path
from time import perf_counter

from snowwi_tools.lib.config_edit import set_config_field, get_config_field
from snowwi_tools.lib import flightline_db as fldb
from snowwi_tools import params


MODE = "snowwi"

# Canonical stage order. `peg` and `output_map` are config-edit steps (no
# external process); the rest shell out to the MIRSL stage scripts.
ALL_STAGES = [
    "setup", "novatel", "peg", "preprocess", "output_map",
    "azmcomp", "postprocess", "estimate_height", "geolocate",
]
_OPTIONAL_STAGES = ("estimate_height", "geolocate")
DEFAULT_STAGES = [s for s in ALL_STAGES if s not in _OPTIONAL_STAGES]

# Canonical region token -> (name aliases, SIGNAL_PARAMS key, DEM filename).
# Aliases cover both DB names (GrandMesa02) and peg IDs (GM2). Camas shares
# Boise's waveform config but keeps its own DEM.
REGIONS = {
    "grandmesa": (("grandmesa", "grand mesa", "gm"), "Grand Mesa", "grand_mesa3.tif"),
    "boise":     (("boise", "bo"),                   "Boise",      "boise2.tif"),
    "camas":     (("camas", "cam"),                  "Boise",      "camas2.tif"),
    # Independent 'test' region. Requires SIGNAL_PARAMS['Test'] in params.py
    # (Pulse length + Receive delay). Set the DEM below to the right test tif.
    "test":      (("test",),                         "Test",       "grand_mesa3.tif"),
}


# --------------------------------------------------------------------------- #
# Region / waveform parameters
# --------------------------------------------------------------------------- #
def _norm(name):
    """Lowercase and strip everything but [a-z0-9] (drops spaces/underscores)."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def region_token(name):
    """Canonical region token ('grandmesa'/'boise'/'camas') for a name/peg id."""
    n = _norm(name)
    for token, (aliases, _sig, _dem) in REGIONS.items():
        if any(n.startswith(_norm(a)) for a in aliases):
            return token
    raise ValueError(
        f"Cannot infer region from name {name!r}. Aliases: "
        f"{ {t: a for t, (a, _s, _d) in REGIONS.items()} }")


def region_for(name):
    """Map a name to (SIGNAL_PARAMS key, DEM filename)."""
    _aliases, sig, dem = REGIONS[region_token(name)]
    return sig, dem


def line_number(name):
    """Trailing integer of a flightline name / peg id (e.g. GrandMesa02 -> 2)."""
    m = re.search(r"(\d+)\s*$", str(name))
    return int(m.group(1)) if m else None


def line_key(name):
    """Canonical (region_token, line_number) key, or None if not derivable.

    Lets DB names and peg ids match across conventions:
    GrandMesa02 <-> GM02 <-> GM2 all key to ('grandmesa', 2).
    """
    try:
        tok = region_token(name)
    except ValueError:
        return None
    num = line_number(name)
    return (tok, num) if num is not None else None


def pulse_length_of(signal_key):
    try:
        return params.SIGNAL_PARAMS[signal_key]["Pulse length"]
    except KeyError as e:
        raise KeyError(
            f"SIGNAL_PARAMS[{signal_key!r}]['Pulse length'] missing") from e


def window_is_zero(window):
    """Reduce a DAQ receive-window string ("21.5us"/"24us"/"0") to a bool.

    Returns True if the window is 0 (calibration), False otherwise. ``None`` /
    blank means "nominal window" -> False. Parses the leading number and
    ignores any unit suffix.
    """
    if window is None:
        return False
    s = str(window).strip()
    if s == "":
        return False
    m = re.match(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", s)
    if not m:
        raise ValueError(f"Cannot parse rx window {window!r} (expect e.g. 21.5us / 0)")
    return float(m.group()) == 0.0


def rx_delay_of(signal_key, band, zero_window):
    """Receive Time Delay (s) for a region/band/window-state from SIGNAL_PARAMS.

    ``zero_window`` selects the ``zero`` (rx window = 0, calibration) vs
    ``nominal`` receive-delay set. Band dependence lives entirely here.
    """
    state = "zero" if zero_window else "nominal"
    try:
        val = params.SIGNAL_PARAMS[signal_key]["Receive delay"][state][band]
    except (KeyError, TypeError) as e:
        raise KeyError(
            f"SIGNAL_PARAMS[{signal_key!r}]['Receive delay'][{state!r}]"
            f"[{band!r}] missing -- expected structure is "
            f"'Receive delay': {{'nominal': {{band: s}}, 'zero': {{band: s}}}}"
        ) from e
    if val is None:
        raise ValueError(
            f"SIGNAL_PARAMS[{signal_key!r}]['Receive delay'][{state!r}]"
            f"[{band!r}] is None -- fill it in before processing.")
    return val


# --------------------------------------------------------------------------- #
# Peg file
# --------------------------------------------------------------------------- #
def parse_peg_file(path):
    """Parse ``pegs_<campaign>.txt`` -> {flightline_id: (lat, lon, hell, hdg)}.

    Header: ``Flightline_ID  PEG_LAT  PEG_LON  PEG_H-ELL  PEG-Heading``.
    """
    pegs = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0].lower().startswith("flightline"):
                continue  # header
            if len(parts) < 5:
                continue
            fid = parts[0]
            lat, lon, hell, hdg = (float(parts[1]), float(parts[2]),
                                   float(parts[3]), float(parts[4]))
            pegs[fid] = (lat, lon, hell, hdg)
    if not pegs:
        raise ValueError(f"No peg rows parsed from {path}")
    return pegs


def match_peg(pegs, flightline):
    """Find the peg row for a flightline, tolerant of naming conventions.

    Tries, in order: exact, case-insensitive, then a canonical
    (region, line-number) key so a DB name (GrandMesa02) matches a peg id
    written as GM2 / GM02.
    """
    if flightline in pegs:
        return pegs[flightline]
    lower = {k.lower(): v for k, v in pegs.items()}
    if flightline.lower() in lower:
        return lower[flightline.lower()]

    target = line_key(flightline)
    if target is not None:
        keyed = {}
        for pid, val in pegs.items():
            k = line_key(pid)
            if k is not None:
                keyed.setdefault(k, []).append((pid, val))
        hits = keyed.get(target, [])
        if len(hits) == 1:
            return hits[0][1]
        if len(hits) > 1:
            raise LookupError(
                f"Ambiguous peg for {flightline!r} (key {target}): "
                f"{[p for p, _ in hits]}")

    raise LookupError(
        f"No peg for flightline {flightline!r} (canonical key "
        f"{line_key(flightline)}). Available IDs: {sorted(pegs)}")


# --------------------------------------------------------------------------- #
# Command / edit helpers (honour --dry-run)
# --------------------------------------------------------------------------- #
class Runner:
    def __init__(self, dry_run=False):
        self.dry_run = dry_run

    def run(self, argv, cwd=None):
        argv = [str(a) for a in argv]
        loc = f" (cwd={cwd})" if cwd else ""
        print(f"  $ {' '.join(argv)}{loc}", flush=True)
        if self.dry_run:
            return
        subprocess.run(argv, cwd=cwd, check=True)

    def edit(self, path, key, value):
        print(f"    edit {os.path.basename(path)}: {key} = {value}",
              flush=True)
        if self.dry_run:
            return
        set_config_field(path, key, value)


# --------------------------------------------------------------------------- #
# Per-flightline pipeline
# --------------------------------------------------------------------------- #
def process_flightline(args, runner, flightline, radar_folder, db_window=None,
                       work_name=None):
    """Run the requested stages for a single flightline.

    ``flightline`` is the line identity (drives region inference + peg match);
    ``work_name`` is the output directory name (defaults to ``flightline``, but
    may be suffixed e.g. ``GrandMesa02_2`` to disambiguate duplicate names).
    """
    work_name = work_name or flightline
    signal_key, dem_file = region_for(flightline)
    work = args.data_root / args.processing_subdir / args.date / work_name
    channels = args.channels

    tag = work_name if work_name == flightline else f"{flightline} -> {work_name}"
    print(f"\n=== {tag}  (bands={' '.join(args.bands)}, region={signal_key}, "
          f"radar={radar_folder}) ===", flush=True)
    print(f"    working dir: {work}", flush=True)

    stages = args.stages
    t_line = perf_counter()

    # setup + novatel run ONCE per flightline (band-agnostic; setup copies all
    # bands' configs, novatel writes inu.h5/flight.kml).
    if "setup" in stages:
        _stage("setup", runner.run, [
            "setup_snowwi_dirs.py", args.date, work_name, radar_folder,
            args.novatel_dir, args.config_dir, "--data-root", args.data_root,
            "--processing-subdir", args.processing_subdir,
        ])

    if "novatel" in stages:
        txt = args.novatel_file or _pick_novatel_txt(work, args.date,
                                                     runner.dry_run)
        _stage("novatel", runner.run,
               ["novatel.py", txt, "-o", "inu.h5"], cwd=work)

    # Everything from peg onward is band-scoped -> loop over the requested bands.
    for band in args.bands:
        _process_band(args, runner, band, work, flightline, signal_key,
                      dem_file, db_window, channels, stages)

    print(f"=== {work_name} done in {perf_counter() - t_line:.1f}s ===",
          flush=True)


def _process_band(args, runner, band, work, flightline, signal_key, dem_file,
                  db_window, channels, stages):
    """Run the band-scoped stages (peg .. geolocate) for one band."""
    band_cfg = work / "config" / band
    azm_cfg = band_cfg / "azmcomp_template.cfg"
    if len(args.bands) > 1:
        print(f"\n  --- band: {band} ---", flush=True)

    # peg + region config ------------------------------------------------- #
    if "peg" in stages:
        print(f"  [peg:{band}] writing peg + region params into config",
              flush=True)
        if not args.no_peg:
            lat, lon, hell, hdg = match_peg(parse_peg_file(args.peg_file),
                                            flightline)
            runner.edit(azm_cfg, "Peg Latitude (degrees)", lat)
            runner.edit(azm_cfg, "Peg Longitude (degrees)", lon)
            runner.edit(azm_cfg, "Peg Heading (degrees)", hdg)
            runner.edit(azm_cfg, "Reference Track Altitude (m)", hell)
        # region waveform params (pulse length + DEM are per-region)
        runner.edit(azm_cfg, "Pulse Length (s)", pulse_length_of(signal_key))
        runner.edit(azm_cfg, "DEM File",
                    str(args.data_root / "dems" / dem_file))
        if args.range_samples is not None:
            runner.edit(azm_cfg, "Number of Range Samples", args.range_samples)
        # Receive Time Delay precedence:
        #   --rx-delay (literal) > window state (--rx-window or DB) -> params.
        # rx delay is per-band, so recomputed for each band here.
        if args.rx_delay is not None:
            rx = args.rx_delay
        else:
            window = args.rx_window if args.rx_window is not None else db_window
            rx = rx_delay_of(signal_key, band, window_is_zero(window))
        for ch in list(channels) + ["tx"]:
            pp_cfg = band_cfg / f"preprocess_{ch}.cfg"
            runner.edit(pp_cfg, "Receive Time Delay (s)", rx)

    # preprocess ---------------------------------------------------------- #
    if "preprocess" in stages:
        _stage(f"preprocess:{band}", runner.run, [
            "preprocess.sh", work, "preprocess", MODE, band, *channels,
        ], cwd=work)

    # optional output-map override ---------------------------------------- #
    if "output_map" in stages and args.output_map is not None:
        nsamp, first, daz = args.output_map
        print(f"  [output_map:{band}] overriding azimuth output map", flush=True)
        runner.edit(azm_cfg, "Number of Azimuth Samples", nsamp)
        runner.edit(azm_cfg, "First Azimuth Sample (m)", first)
        runner.edit(azm_cfg, "Azimuth Spacing (m)", daz)

    # azmcomp ------------------------------------------------------------- #
    if "azmcomp" in stages:
        _stage(f"azmcomp:{band}", runner.run, [
            "azmcomp.sh", work, "azmcomp", "preprocess", MODE, band,
            args.patch_size, str(args.swap_channels), *channels,
        ], cwd=work)

    # postprocess --------------------------------------------------------- #
    if "postprocess" in stages:
        _stage(f"postprocess:{band}", runner.run, [
            "postprocess.sh", work, "postprocess", "azmcomp", MODE, band,
            args.azm_looks, args.rng_looks, *channels,
        ], cwd=work)

    # estimate_height (optional) ------------------------------------------ #
    if "estimate_height" in stages:
        _stage(f"estimate_height:{band}", runner.run, [
            "estimate_height.sh", work, "estimate_height", "postprocess",
            MODE, band, args.azm_looks, args.rng_looks, *channels,
            "-l", args.ml_looks[0], args.ml_looks[1],
        ], cwd=work)

    # geolocate (optional) ------------------------------------------------ #
    # Per channel, in the postprocess channel dir. geolocate.py needs the DEM
    # (height) at the SAME resolution as the data being geolocated:
    #   full res  -> height from azmcomp.h5     (-r = az/rng spacing)
    #   multilook -> height from data_ml.h5     (-r = spacing * look factor)
    # Always produces the two geolocated SLC products; with --power it also
    # makes the two log-power (RCS dB, via slc_to_mag.py) products -> 4 total.
    if "geolocate" in stages:
        print(f"  [geolocate:{band}]", flush=True)
        t = perf_counter()
        az_sp, rng_sp = _geo_spacing(args, azm_cfg)
        azl, rnl = int(args.azm_looks), int(args.rng_looks)
        full_r = (az_sp, rng_sp)
        ml_r = (az_sp * azl, rng_sp * rnl)
        mll = f"{args.azm_looks}_{args.rng_looks}"
        for ch in channels:
            ch_dir = work / "postprocess" / band / str(ch)
            cfg = f"snowwi_azmcomp_{ch}.cfg"
            # full-res + ML calibrated SLC, geolocated
            _geolocate(runner, ch_dir, cfg, "azmcomp.h5",
                       "slc_calibrated.h5", "slc_distributed",
                       "slc_cal_utm", args.geo_looks, full_r)
            _geolocate(runner, ch_dir, cfg, "data_ml.h5",
                       f"slc_ml_{mll}.h5", "value",
                       "slc_ml_utm", args.geo_looks, ml_r)
            if args.power:
                # full-res power (RCS dB)
                runner.run(["slc_to_mag.py", "slc_calibrated.h5", "-p", "-lp",
                            "-o", "slc_mag_log.h5"], cwd=ch_dir)
                _geolocate(runner, ch_dir, cfg, "azmcomp.h5",
                           "slc_mag_log.h5", "magnitude",
                           "slc_mag_log_utm", args.geo_looks, full_r)
                # ML power (RCS dB)
                runner.run(["slc_to_mag.py", f"slc_cal_ml_{mll}.h5", "-p", "-lp",
                            "-o", "slc_cal_ml_mag_log.h5"], cwd=ch_dir)
                _geolocate(runner, ch_dir, cfg, "data_ml.h5",
                           "slc_cal_ml_mag_log.h5", "magnitude",
                           "slc_cal_ml_mag_log_utm", args.geo_looks, ml_r)
        print(f"    geolocate:{band}: {perf_counter() - t:.1f}s", flush=True)


def _stage(name, run, argv, cwd=None):
    print(f"  [{name}]", flush=True)
    t = perf_counter()
    run(argv, cwd=cwd)
    print(f"    {name}: {perf_counter() - t:.1f}s", flush=True)


def _geo_spacing(args, azm_cfg):
    """(az, rng) pixel spacing for geolocate -r: --geo-res override, else the
    Azimuth/Range Spacing from the azmcomp config; falls back to (1.0, 1.0)."""
    if args.geo_res is not None:
        return float(args.geo_res[0]), float(args.geo_res[1])
    az = get_config_field(azm_cfg, "Azimuth Spacing (m)")
    rg = get_config_field(azm_cfg, "Range Spacing (m)")
    try:
        return float(az), float(rg)
    except (TypeError, ValueError):
        return 1.0, 1.0


def _geolocate(runner, cwd, cfg, height_file, data_file, data_ds, out_name,
               looks, res):
    """One geolocate.py invocation (height + data -> UTM raster)."""
    runner.run([
        "geolocate.py", cfg,
        "--looks", looks[0], looks[1],
        "--height", height_file, "height",
        "-d", data_file, data_ds, "0", out_name,
        "-u", "-r", f"{res[0]}", f"{res[1]}", "-t", "-v",
    ], cwd=cwd)


def _pick_novatel_txt(work, date, dry_run):
    """Pick the single ``*DATE*.txt`` copied into the working dir by setup."""
    matches = sorted(glob.glob(str(work / f"*{date}*.txt")))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        if dry_run:
            return str(work / f"<DATE>.txt")  # placeholder for dry-run print
        raise FileNotFoundError(
            f"No *{date}*.txt in {work}; run setup first or pass --novatel-file")
    raise ValueError(
        f"Multiple Novatel .txt in {work}: {matches}. Pass --novatel-file.")


# --------------------------------------------------------------------------- #
# Whole-date mode
# --------------------------------------------------------------------------- #
def process_date(args, runner):
    conn = fldb.connect_db(args.flightline_db)
    rows = fldb.rows_for_date(conn, args.date)
    if not rows:
        sys.exit(f"No flightlines for date {args.date} in {args.flightline_db}")
    print(f"Date {args.date}: {len(rows)} flightline(s) from DB", flush=True)

    failures = []
    seen = {}
    for row in rows:
        fl = row["flightline_name"]
        folder = row["folder_name"]
        db_window = fldb.rx_window_of(row)
        # Disambiguate duplicate flightline names: first keeps its name, the
        # 2nd/3rd get _2/_3 appended to the output dir (identity stays `fl`).
        seen[fl] = seen.get(fl, 0) + 1
        work_name = fl if seen[fl] == 1 else f"{fl}_{seen[fl]}"
        try:
            process_flightline(args, runner, fl, folder, db_window=db_window,
                               work_name=work_name)
        except Exception as e:  # continue-on-error across the batch
            print(f"!!! {work_name} FAILED: {e}", flush=True)
            failures.append((work_name, str(e)))

    print("\n===== date summary =====", flush=True)
    ok = len(rows) - len(failures)
    print(f"  {ok}/{len(rows)} succeeded", flush=True)
    for fl, err in failures:
        print(f"  FAILED {fl}: {err}", flush=True)
    if failures:
        sys.exit(1)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("date", help="Flight date YYYYMMDD")
    p.add_argument("flightline", nargs="?",
                   help="Flightline name (omit to process the whole date)")
    p.add_argument("-b", "--band", dest="bands", required=True, nargs="+",
                   choices=["low", "high", "c"], metavar="BAND",
                   help="Band(s) to process, e.g. -b low  or  -b high c")
    p.add_argument("-c", "--channels", nargs="+", default=["0", "2"],
                   help="Channels (default: 0 2)")

    p.add_argument("--peg-file", help="pegs_<campaign>.txt (peg source)")
    p.add_argument("--no-peg", action="store_true",
                   help="Do not touch the peg (assume config is already set)")

    p.add_argument("--flightline-db",
                   help="Flightline sqlite DB (path or s3://…); required for "
                        "date mode and for name->folder lookup")
    p.add_argument("--radar-data",
                   help="Radar dataset folder (YYYYMMDDThhmmss) or abs path; "
                        "overrides DB lookup in single-flightline mode")

    p.add_argument("--data-root", default=str(Path.home() / "data"),
                   help="Data root (default: ~/data)")
    p.add_argument("--processing-subdir", default="processing",
                   help="Output subdir under data-root (default: processing). "
                        "Use e.g. processing_v2 to avoid overwriting.")
    p.add_argument("--config-dir",
                   default=str(Path.home() / "data" / "configs" / "most_recent"),
                   help="Config template dir with low/ high/ c/ subdirs")
    p.add_argument("--novatel-dir",
                   default=str(Path.home() / "data" / "novatel"),
                   help="Dir of raw Novatel .txt files (for setup)")
    p.add_argument("--novatel-file",
                   help="Explicit Novatel .txt for the novatel stage")

    p.add_argument("--range-samples", type=int,
                   help="Override Number of Range Samples in azmcomp")
    p.add_argument("--output-map", nargs=3, metavar=("NSAMP", "FIRST", "DAZ"),
                   help="Override azimuth output map (post-preprocess)")
    p.add_argument("--rx-delay",
                   help="Force literal Receive Time Delay (s) for all channels "
                        "(highest precedence; bypasses SIGNAL_PARAMS)")
    p.add_argument("--rx-window",
                   help="Force the DAQ receive window (e.g. 21.5us / 24us / 0) "
                        "instead of the DB value; selects nominal vs zero "
                        "receive-delay set from SIGNAL_PARAMS")

    p.add_argument("--azm-looks", default="4", help="Azimuth looks (default 4)")
    p.add_argument("--rng-looks", default="4", help="Range looks (default 4)")
    p.add_argument("--ml-looks", nargs=2, default=["10", "10"],
                   metavar=("ML_AZ", "ML_RNG"),
                   help="Extra multilook for estimate_height (default 10 10)")
    p.add_argument("--patch-size", type=int, default=4000,
                   help="azmcomp patch size (default 4000)")
    p.add_argument("--swap-channels", action="store_true",
                   help="Pass swap_channels=True to azmcomp.sh")
    p.add_argument("--geo-looks", nargs=2, default=["1", "1"],
                   metavar=("AZ", "RNG"),
                   help="geolocate.py --looks (default 1 1; inputs are already "
                        "at their grid)")
    p.add_argument("--geo-res", nargs=2, default=None,
                   metavar=("AZ", "RNG"),
                   help="Base pixel spacing for geolocate -r. Default: read "
                        "Azimuth/Range Spacing from the azmcomp config (ML "
                        "products scale it by the look factors).")
    p.add_argument("--power", action="store_true",
                   help="Also geolocate log-power (RCS dB) products via "
                        "slc_to_mag.py -> 4 products per channel instead of 2")

    p.add_argument("-s", "--stages", default=",".join(DEFAULT_STAGES),
                   help=f"Comma-separated stages (default: no heights/geolocate). "
                        f"Choices: {','.join(ALL_STAGES)}")
    p.add_argument("--with-heights", action="store_true",
                   help="Append estimate_height to the default stages")
    p.add_argument("--with-geolocate", action="store_true",
                   help="Append geolocate to the default stages")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands and edits without executing")

    args = p.parse_args()

    # normalise
    args.data_root = Path(args.data_root).expanduser().resolve()
    args.stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    if args.with_heights and "estimate_height" not in args.stages:
        args.stages.append("estimate_height")
    if args.with_geolocate and "geolocate" not in args.stages:
        args.stages.append("geolocate")
    bad = set(args.stages) - set(ALL_STAGES)
    if bad:
        p.error(f"Unknown stage(s): {sorted(bad)}. Choices: {ALL_STAGES}")
    if args.output_map is not None:
        args.output_map = (int(args.output_map[0]), int(args.output_map[1]),
                           float(args.output_map[2]))
    if "peg" in args.stages and not args.no_peg and not args.peg_file:
        p.error("peg stage needs --peg-file (or use --no-peg)")
    if args.flightline is None and not args.flightline_db:
        p.error("date mode (no FLIGHTLINE) needs --flightline-db")
    return args


def main():
    args = parse_args()
    runner = Runner(dry_run=args.dry_run)

    if args.flightline is None:
        process_date(args, runner)
        return

    # single flightline: resolve radar folder
    db_window = None
    radar = args.radar_data
    if radar is None:
        if not args.flightline_db:
            sys.exit("Single flightline needs --radar-data or --flightline-db")
        conn = fldb.connect_db(args.flightline_db)
        row = fldb.lookup_flightline(conn, args.flightline, args.date)
        radar = row["folder_name"]
        db_window = fldb.rx_window_of(row)

    process_flightline(args, runner, args.flightline, radar,
                       db_window=db_window)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
