#!/usr/bin/env python3
"""Set up the directory structure for processing a single SAR flightline.

Creates processing/DATE/FLIGHTLINE/, copies config templates for all three
bands (low, high, c), creates per-channel azmcomp symlinks, links patterns
and radar data, and copies all Novatel .txt files matching the flight date.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path


BANDS = ("low", "high", "c")


def parse_args():
    p = argparse.ArgumentParser(
        description="Set up symlinks and copy files for a SAR flightline processing directory."
    )
    p.add_argument("date", metavar="DATE", help="Flight date YYYYMMDD")
    p.add_argument("flightline", metavar="FLIGHTLINE", help="Flightline name, e.g. GrandMesa2")
    p.add_argument(
        "radar_data",
        metavar="RADAR_DATA",
        help=(
            "Radar data directory: timestamp (e.g. 20260128T211011) resolved under "
            "data/radar_data/DATE/, or an absolute path"
        ),
    )
    p.add_argument(
        "novatel_dir",
        metavar="NOVATEL_DIR",
        help="Directory containing Novatel .txt files; all files matching *DATE*.txt are copied",
    )
    p.add_argument(
        "config_dir",
        metavar="CONFIG_DIR",
        help="Config template directory; must contain low/, high/, and c/ subdirs",
    )
    p.add_argument(
        "--data-root",
        default=str(Path.home() / "data"),
        help="Data root directory (default: ~/data)",
    )
    p.add_argument(
        "--processing-subdir",
        default="processing",
        help="Output subdirectory under data-root (default: processing). "
             "Use e.g. processing_v2 to avoid overwriting existing runs.",
    )
    return p.parse_args()


def resolve_radar_data(data_root: Path, date: str, radar_data: str) -> Path:
    rd = Path(radar_data)
    if rd.is_absolute():
        return rd
    return data_root / "radar_data" / date / radar_data


def check_exists(path: Path, label: str) -> None:
    if not path.exists():
        sys.exit(f"Not found ({label}): {path}")


def make_rel_symlink(link: Path, target: Path) -> None:
    rel = Path(os.path.relpath(target, link.parent))
    link.symlink_to(rel)
    print(f"  {link.name} -> {rel}")


def setup(args) -> None:
    data_root = Path(args.data_root).resolve()

    config_root = Path(args.config_dir).resolve()
    for band in BANDS:
        check_exists(config_root / band, f"config template {config_root}/{band}")

    patterns_dir = data_root / "patterns"
    check_exists(patterns_dir, "patterns dir")

    radar_data_path = resolve_radar_data(data_root, args.date, args.radar_data)
    check_exists(radar_data_path, "radar data dir")

    novatel_dir = Path(args.novatel_dir).resolve()
    check_exists(novatel_dir, "novatel dir")
    novatel_files = sorted(novatel_dir.glob(f"*{args.date}*.txt"))
    if not novatel_files:
        sys.exit(f"No Novatel .txt files matching *{args.date}*.txt in {novatel_dir}")

    # Create processing directory
    proc_dir = data_root / args.processing_subdir / args.date / args.flightline
    proc_dir.mkdir(parents=True, exist_ok=True)
    print(f"Processing dir: {proc_dir}")

    # Copy config templates for each band
    for band in BANDS:
        src_band = config_root / band
        dst_band = proc_dir / "config" / band
        dst_band.mkdir(parents=True, exist_ok=True)

        for src_file in sorted(src_band.iterdir()):
            if not src_file.is_file():
                continue
            if src_file.stem.startswith("snowwi_azmcomp_"):
                continue
            dst_file = dst_band / src_file.name
            if dst_file.exists():
                print(f"  skip existing: config/{band}/{src_file.name}")
                continue
            shutil.copy2(src_file, dst_file)

        # Normalise main azmcomp template name
        legacy = dst_band / "snowwi_azmcomp.cfg"
        template = dst_band / "azmcomp_template.cfg"
        if legacy.exists() and not template.exists():
            legacy.rename(template)
            print(f"  renamed snowwi_azmcomp.cfg -> azmcomp_template.cfg in config/{band}/")

        # Per-channel symlinks -> azmcomp_template.cfg
        if template.exists():
            for i in range(4):
                link = dst_band / f"snowwi_azmcomp_{i}.cfg"
                if link.exists() or link.is_symlink():
                    continue
                link.symlink_to("azmcomp_template.cfg")
            print(f"  config/{band}/ ready (snowwi_azmcomp_{{0..3}}.cfg -> azmcomp_template.cfg)")
        else:
            print(f"  config/{band}/ ready (no azmcomp_template.cfg found, skipped channel links)")

    # patterns symlink (relative)
    patterns_link = proc_dir / "patterns"
    if not patterns_link.exists() and not patterns_link.is_symlink():
        make_rel_symlink(patterns_link, patterns_dir)

    # full -> radar data path, radar_data -> full
    full_link = proc_dir / "full"
    if not full_link.exists() and not full_link.is_symlink():
        if radar_data_path.is_absolute():
            full_link.symlink_to(radar_data_path)
            print(f"  full -> {radar_data_path}")
        else:
            make_rel_symlink(full_link, radar_data_path)

    radar_data_link = proc_dir / "radar_data"
    if not radar_data_link.exists() and not radar_data_link.is_symlink():
        radar_data_link.symlink_to("full")
        print("  radar_data -> full")

    # Copy Novatel .txt files
    for novatel_txt in novatel_files:
        dst_txt = proc_dir / novatel_txt.name
        if not dst_txt.exists():
            shutil.copy2(novatel_txt, dst_txt)
            print(f"  copied {novatel_txt.name}")
        else:
            print(f"  skip existing: {novatel_txt.name}")

    print(f"\nDone. {proc_dir}")


def main():
    args = parse_args()
    setup(args)


if __name__ == "__main__":
    main()
