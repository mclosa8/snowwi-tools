"""
    UAVSAR-related functions for quality of life.

    Author(s): Marc Closa Tarres (MCT)
    Date: 2025-09-03
    Version: v1

    Changelog:
        - v0: Initial version. Sept 03, 2025 - MCT
        - v1: Added functions to open GRD and SCH UAVSAR filetypes. Sept 04, 2025 - MCT
        - v2: Added functions to rasterize scenes. Sept 09, 2025 - MCT
"""
import numpy as np

from osgeo import gdal, osr
import os
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


def open_uavsar_binary(fname, n_cols, format=np.float32, nan=-10000.0):
    samps = np.fromfile(fname, dtype=format).reshape((-1, n_cols))
    samps[samps==nan] = np.nan
    print(samps.shape)
    return samps


def open_uavsar_grd_files(data_fname, nan=-10000.0, open_prc=True):
    filename, product_coords = os.path.splitext(data_fname) # filename will be sth like /a/long/path/to/flightline_name.product
    filename = filename.split('/')[-1] # Now flightline_name.product_type
    flightline, product_type = os.path.splitext(filename)

    ann_filename = data_fname.replace(product_type + product_coords, '.ann')
    prc_hgt_fname = data_fname.replace(product_type, '.prc')

    print("Opening GLISTIN GRD files:")
    print(f"    Data file: {data_fname}")
    print(f"    Ann file: {ann_filename}")
    
    if open_prc:
        assert os.path.isfile(prc_hgt_fname)
        print(f"    Prc file: {prc_hgt_fname}")
    
    ann_dict = parse_uavsar_annotation_file(ann_filename, product_ext=product_coords)['flat']

    grd_spacing = (
        ann_dict['grd_latitude_spacing'],
        ann_dict['grd_longitude_spacing']
    )

    grd_samps = (
        ann_dict['grd_latitude_lines'],
        ann_dict['grd_longitude_samples']
    ) # Latitude samples, longitude samples

    ul = (
        ann_dict['grd_starting_latitude'],
        ann_dict['grd_starting_longitude']
    )

    lr = (
        ul[0] + grd_samps[0] * grd_spacing[0],
        ul[1] + grd_samps[1] * grd_spacing[1]
    )
    ullr = (ul, lr)

    grd_params = {
         "ullr": ullr,
         "samples": grd_samps,
         "spacing": grd_spacing
    }

    data = open_uavsar_binary(data_fname, grd_samps[1])

    if open_prc:
        prc = open_uavsar_binary(prc_hgt_fname, grd_samps[1])
        return flightline, data, grd_params, prc

    return flightline, data, grd_params


def open_uavsar_sch_files(data_fname, nan=-10000.0):
    filename, product_coords = os.path.splitext(data_fname) # filename will be sth like /a/long/path/to/flightline_name.product
    filename = filename.split('/')[-1] # Now flightline_name.product_type
    flightline, product_type = os.path.splitext(filename)

    ann_filename = data_fname.replace(product_type+'.sch', '.ann')

    print("Opening GLISTIN SCH files:")
    print(f"    Data file: {data_fname}")
    print(f"    Ann file: {ann_filename}")
    
    ann_dict = parse_uavsar_annotation_file(ann_filename, product_ext=product_coords)['flat']

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


def array2raster(raster_fname, raster_origin, pixel_resolution, data):
    """
        Converts array into raster file using GDAL.
        Source: https://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html#create-raster-from-array

        Inputs:
        -------
            - raster_fname: Output raster filename
            - raster_origin: Upper-left (lat, lon) in degrees.
            - pixel_resolution: (lat, lon) pixel resolution in degrees.
            - data: Array to convert.
    """
    rows, cols = data.shape

    origin_lat, origin_lon = raster_origin

    res_y, res_x = pixel_resolution

    gtrans = (
        origin_lon,     # Origin X
        res_x,          # Pixel resolution X (pixel width)
        0,
        origin_lat,     # Origin Y
        0,
        res_y           # Pixel resolution Y (pixel height)
    )

    driver = gdal.GetDriverByName('GTiff')
    out_raster = driver.Create(raster_fname, cols, rows, 1, gdal.GDT_Byte)
    out_raster.SetGeoTransform(gtrans)
    out_band = out_raster.GetRasterBand(1)
    out_band.WriteArray(data)
    out_raster_srs = osr.SpatialReference()
    out_raster_srs.ImportFromEPSG(4326)
    out_raster.SetProjection(out_raster_srs.ExportToWkt())
    out_band.FlushCache()


def crop_to_bbox(_pass, bbox_ullr, buffer=0):
    
    data = _pass['data']
    params = _pass['params']
    if 'prc' in _pass.keys():
        prc = True
        prc_data = _pass['prc']
    else:
        prc = False

    print("Before: ", data.shape)

    # Now make lat,lon vectors
    lat = np.linspace(params['ullr'][0][0], params['ullr'][1][0], params['samples'][0])
    lon = np.linspace(params['ullr'][0][1], params['ullr'][1][1], params['samples'][1])

    assert abs(lat.min() - params['ullr'][1][0]) < 1e-9, "Incorrect lat vector"
    assert abs(lat.max() - params['ullr'][0][0]) < 1e-9, "Incorrect lat vector"
    assert abs(lon.min() - params['ullr'][0][1]) < 1e-9, "Incorrect lon vector"
    assert abs(lon.max() - params['ullr'][1][1]) < 1e-9, "Incorrect lon vector"

    assert data.shape == (lat.size, lon.size), "Incorrect data or lat/lon shape"

    row_idxs = np.where((bbox_ullr[1][0] <= lat) & (lat <= bbox_ullr[0][0]))[0]
    col_idxs = np.where((bbox_ullr[0][1] <= lon ) & (lon <= bbox_ullr[1][1]))[0]

    cropped = data[row_idxs[0]:row_idxs[-1], col_idxs[0]:col_idxs[-1]]
    print("After: ", cropped.shape)

    if prc:
        cropped_prc = prc_data[row_idxs[0]:row_idxs[-1], col_idxs[0]:col_idxs[-1]]
        assert cropped.shape == cropped_prc.shape, "Shape missmatch between hgt data and prc data."

        return {
            'data': cropped,
            'prc': cropped_prc,
            'ullr': bbox_ullr,
            'spacing': np.array(params['spacing'])
        }        

    return {
        'data': cropped,
        'ullr': bbox_ullr,
        'spacing': np.array(params['spacing'])
    }


def get_bbox_flightline(passes):
    bbox_ullr = np.empty((2, 2)) 

    """
        bbox_ullr = (
            (ul_lat, ul_lon),
            (lr_lat, lr_lon)
        )
    """

    for i, pass_key in enumerate(passes.keys()):
        _pass = passes[pass_key]
        if i == 0:
            bbox_ullr = np.array(_pass['params']['ullr'])
        else:
            ullr_pass = np.array(_pass['params']['ullr'])
            ul = np.array(
                (np.min((bbox_ullr[0][0], ullr_pass[0][0])),
                np.max((bbox_ullr[0][1], ullr_pass[0][1]))),
            )
            lr = np.array(
                (np.max((bbox_ullr[1][0], ullr_pass[1][0])),
                np.min((bbox_ullr[1][1], ullr_pass[1][1]))),
            )
            bbox_ullr[0] = ul
            bbox_ullr[1] = lr

    print(bbox_ullr)
    
    return bbox_ullr


def transform_to_epsg(ds, dst_epsg=4326):
    # Source geotransform
    gt = ds.GetGeoTransform()
    
    # Build spatial references
    src = osr.SpatialReference()
    src.ImportFromWkt(ds.GetProjection())

    dst = osr.SpatialReference()
    dst.ImportFromEPSG(dst_epsg)

    transform = osr.CoordinateTransformation(src, dst)

    cols = np.arange(ds.RasterXSize)
    rows = np.arange(ds.RasterYSize)

    xx, yy, = np.meshgrid(cols, rows)

    x = gt[0] + xx * gt[1] + yy * gt[2]
    y = gt[3] + xx * gt[4] + yy * gt[5]

    coords = np.vstack((x.ravel(), y.ravel())).T

    lonlat = transform.TransformPoints(coords)

    lon = np.array([pt[0] for pt in lonlat]).reshape(xx.shape)
    lat = np.array([pt[1] for pt in lonlat]).reshape(xx.shape)

    return lat, lon

if __name__ == "__main__":
    # TODO: improve argument handling
    ann_text = sys.argv[1]
    parsed = parse_uavsar_annotation(ann_text, snake_case_keys=True)

    # Examples:
    print(parsed["flat"]["uavsar_annotation_type"])                   # -> "TOPSAR"
    print(parsed["sections"]["GRD Product Information"]["grd_latitude_lines"])  # -> 10457
    print(parsed["units"]["grd_latitude_spacing"])                    # -> "deg"
    print(parsed["mdx_commands"].get("SCH Height"))                   # -> "mdx -s 5166 ..."
