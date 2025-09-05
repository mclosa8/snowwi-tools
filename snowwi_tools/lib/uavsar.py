"""
    UAVSAR-related functions for quality of life.

    Author(s): Marc Closa Tarres (MCT)
    Date: 2025-09-03
    Version: v1

    Changelog:
        - v0: Initial version. Sept 03, 2025 - MCT
        - v1: Added functions to open GRD and SCH UAVSAR filetypes. Sept 04, 2025 - MCT
"""
import numpy as np

import os
import rasterio
import re
import sys

from typing import Any, Dict, Optional

from pprint import PrettyPrinter
pp = PrettyPrinter()

_KV_RE = re.compile(
    r'^\s*(?P<key>.+?)\s*\(\s*(?P<unit>[^)]*?)\s*\)\s*=\s*(?P<value>.*?)\s*(?:;.*)?$'
)
_SECTION_BAR = re.compile(r'^\s*;\s*-{3,}\s*$')
_SECTION_TITLE = re.compile(r'^\s*;\s*(?P<title>[^;].*?)\s*$')


def _snake(s: str) -> str:
    s = re.sub(r"[^\w]+", "_", s.strip()).strip("_").lower()
    return re.sub(r"_+", "_", s)


def _to_number(tok: str):
    try:
        if re.fullmatch(r"[+-]?\d+", tok):
            return int(tok)
        return float(tok)
    except Exception:
        return None


def _parse_value(v: str) -> Any:
    v = v.strip()
    if not v or v.upper() == "N/A":
        return None

    # AxB like "2x4" -> [2,4]
    axb = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", v)
    if axb:
        return [int(axb.group(1)), int(axb.group(2))]

    # If it's a URL or clearly text, keep as string
    if "http://" in v or "https://" in v:
        return v

    # Try list of numbers ONLY if the whole string is numbers separated by spaces
    tokens = v.split()
    if tokens and all(_to_number(t) is not None for t in tokens):
        nums = [_to_number(t) for t in tokens]
        return nums[0] if len(nums) == 1 else nums

    # Try single number
    n = _to_number(v)
    return n if n is not None else v


def open_annotation(ann_file):
    try:
        with open(ann_file, 'r') as f:
            contents = f.read()
    except FileNotFoundError:
        print(".ann file not found")
        return 0
    return contents


def parse_uavsar_annotation(
    text: str,
    *,
    product_ext: Optional[str] = None,   # ".grd"/"grd" or ".sch"/"sch"; None = no filtering
    snake_case_keys: bool = True,
    debug: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Parse a UAVSAR .ann **text** (contents) into dicts.
    If product_ext is provided ('grd' or 'sch'), keep ALL other sections,
    but:
      - keep only the matching '<PROD> Product Information' section, and
      - within 'List of Product Data Files', keep only keys for that product.
    Returns:
      {
        "sections": {section_name: {key: value, ...}, ...},
        "flat":     {key: value, ...},
        "units":    {key: unit_str, ...},
      }
    """
    sections: Dict[str, Dict[str, Any]] = {}
    units_all: Dict[str, str] = {}
    flat_all: Dict[str, Any] = {}

    current_section = "Header"

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        # print(f"{i}/{len(lines)}")
        raw = lines[i]
        # normalize odd whitespace
        line = raw.replace("\t", " ").replace("\xa0", " ").rstrip()
        stripped = line.strip()

        # ----- Comment / header handling -----
        if stripped.startswith(";"):
            # Strict section header: BAR -> TITLE -> BAR
            if _SECTION_BAR.match(stripped):
                if i + 2 < len(lines):
                    nxt = lines[i + 1].strip()
                    nxt2 = lines[i + 2].strip()
                    m_title = _SECTION_TITLE.match(nxt)
                    if m_title and _SECTION_BAR.match(nxt2):
                        current_section = m_title.group("title").strip()
                        sections.setdefault(current_section, {})
                        i += 3  # consume bar, title, bar
                        continue
                # lone bar; consume and continue
                i += 1
                continue

            # Other comment lines: ignore (do NOT switch sections)
            i += 1
            continue

        # ----- Blank lines -----
        if not stripped:
            i += 1
            continue

        # ----- Key/Unit/Value lines -----
        m = _KV_RE.match(line)
        if m:
            key = m.group("key").strip()
            unit = (m.group("unit") or "").strip()
            value = _parse_value(m.group("value"))

            k = _snake(key) if snake_case_keys else key
            sections.setdefault(current_section, {})[k] = value
            flat_all[k] = value
            units_all[k] = unit
        elif debug:
            print(f"[DEBUG] No KV match for line {i+1}: {repr(line)}")

        i += 1

    return {"sections": sections, "flat": flat_all, "units": units_all}


def parse_uavsar_annotation_file(path: str, **kwargs) -> Dict[str, Dict[str, Any]]:
    """Convenience: read from file path and parse."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return parse_uavsar_annotation(f.read(), **kwargs)


def open_uavsar_binary(fname, n_cols, format=np.float32):
    samps = np.fromfile(fname, dtype=format).reshape((-1, n_cols))
    print(samps.shape)
    return samps


def open_uavsar_grd_files(data_fname):
    filename, product_coords = os.path.splitext(data_fname) # filename will be sth like /a/long/path/to/flightline_name.product
    filename = filename.split('/')[-1] # Now flightline_name.product_type
    flightline, product_type = os.path.splitext(filename)

    ann_filename = data_fname.replace(product_type+product_coords, '.ann')

    print("Opening GLISTIN GRD files:")
    print(f"    Data file: {data_fname}")
    print(f"    Ann file: {ann_filename}")
    
    ann_dict = parse_uavsar_annotation_file(ann_filename, product_ext=product_coords)['flat']

    ul = (
        ann_dict['approximate_upper_left_latitude'],
        ann_dict['approximate_upper_left_longitude']
    )

    lr = (
        ann_dict['approximate_lower_right_latitude'],
        ann_dict['approximate_lower_right_longitude']
    )

    ullr = (ul, lr)

    grd_spacing = (
        ann_dict['grd_latitude_spacing'],
        ann_dict['grd_longitude_spacing']
    )

    grd_samps = (
        ann_dict['grd_latitude_lines'],
        ann_dict['grd_longitude_samples']
    ) # Latitude samples, longitude samples

    grd_params = {
         "ullr": ullr,
         "samples": grd_samps,
         "spacing": grd_spacing
    }

    data = open_uavsar_binary(data_fname, grd_samps[1])

    return flightline, data, grd_params


def open_uavsar_sch_files(data_fname):
    filename, product_coords = os.path.splitext(data_fname) # filename will be sth like /a/long/path/to/flightline_name.product
    filename = filename.split('/')[-1] # Now flightline_name.product_type
    flightline, product_type = os.path.splitext(filename)

    ann_filename = data_fname.replace(product_type+'.sch', '.ann')

    print("Opening GLISTIN SCH files:")
    print(f"    Data file: {data_fname}")
    print(f"    Ann file: {ann_filename}")
    
    ann_dict = parse_uavsar_annotation_file(ann_filename, product_ext=product_coords)['flat']
    pp.pprint(ann_dict)

    peg = (
        ann_dict['peg_latitude'],
        ann_dict['peg_longitude'],
        ann_dict['peg_heading'],
        ann_dict['peg_radius_of_curvature']
    ) # Lat (deg), lon (deg), heading (deg), radius of curvature (m)

    first_sample = (
        ann_dict['sch_along_track_distance_from_peg_to_first_line'],
        ann_dict['sch_cross_track_distance_from_peg_to_first_sample']
    ) # From PEG point: (along track, cross_track) (m)

    peg_spacing = (
        ann_dict['sch_along_track_line_spacing'],
        ann_dict['sch_cross_track_sample_spacing']
    ) # Along track resolution, cross-track resolution (m)

    peg_samples = (
        ann_dict['sch_number_of_along_track_samples'],
        ann_dict['sch_number_of_cross_track_samples']
    ) # Along-track, cross-track (samps)

    data = open_uavsar_binary(data_fname, peg_samples[0])

    sch_params = {
        "peg_point": peg,
        "first_sample": first_sample,
        "spacing": peg_spacing,
        "samples": peg_samples
    }

    return flightline, data, sch_params


def uavsar_to_tiff(data, lat, lon):
    shape = data.shape

    lat_tile = 

if __name__ == "__main__":
    # TODO: improve argument handling
    ann_text = sys.argv[1]
    parsed = parse_uavsar_annotation(ann_text, snake_case_keys=True)

    # Examples:
    print(parsed["flat"]["uavsar_annotation_type"])                   # -> "TOPSAR"
    print(parsed["sections"]["GRD Product Information"]["grd_latitude_lines"])  # -> 10457
    print(parsed["units"]["grd_latitude_spacing"])                    # -> "deg"
    print(parsed["mdx_commands"].get("SCH Height"))                   # -> "mdx -s 5166 ..."
