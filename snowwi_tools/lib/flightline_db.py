"""Read-only access to the SNOWWI flightline sqlite database.

The database is created and maintained interactively by
``bin/flightline_db_manager.py``. This module only *reads* it, for the
``process_snowwi.py`` driver, which needs two mappings:

* ``flightline_name`` -> ``folder_name`` (the ``YYYYMMDDThhmmss`` radar-data
  directory) for a single flightline, and
* all flightlines flown on a given ``flight_date`` (for whole-date processing).

It also exposes the optional per-line ``rx_window`` column: the DAQ receive
window used for that line (e.g. ``"21.5us"`` / ``"24us"`` / ``"0"``). The
driver reads it only as binary (window == 0 -> calibration) to choose between
the ``nominal`` and ``zero`` receive-delay sets in ``SIGNAL_PARAMS``.

The sqlite path may be local or an ``s3://bucket/key`` URI; for S3 the file is
downloaded to a temp file and opened read-only (nothing is written back).
"""

import os
import sqlite3
import tempfile
from urllib.parse import urlparse


TABLE = "flightlines"


def is_s3_path(path):
    return str(path).startswith("s3://")


def _parse_s3_path(s3_path):
    parsed = urlparse(s3_path)
    return parsed.netloc, parsed.path.lstrip("/")


def connect_db(db_path):
    """Open the flightline DB (local path or ``s3://…``) read-only.

    Returns a ``sqlite3.Connection`` with ``row_factory = sqlite3.Row`` so
    callers can access columns by name. For S3 paths the object is downloaded
    to a temp file first; the temp file is left in place for the life of the
    process (small; OS cleans ``/tmp``).
    """
    if is_s3_path(db_path):
        import boto3  # lazy: only needed for S3

        bucket, key = _parse_s3_path(db_path)
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        boto3.client("s3").download_fileobj(bucket, key, tmp)
        tmp.flush()
        tmp.close()
        local_path = tmp.name
    else:
        local_path = os.fspath(db_path)
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Flightline DB not found: {local_path}")

    conn = sqlite3.connect(local_path)
    conn.row_factory = sqlite3.Row
    return conn


def _columns(conn):
    cur = conn.execute(f"PRAGMA table_info({TABLE})")
    return [row[1] for row in cur.fetchall()]


def _has_rx_delay(conn):
    return "rx_delay" in _columns(conn)


def _date_variants(date):
    """Return plausible string forms of a date for tolerant matching.

    Accepts ``YYYYMMDD`` (the driver's canonical form) or an already-dashed
    form and yields both ``YYYYMMDD`` and ``YYYY-MM-DD`` (and slashed).
    """
    digits = str(date).replace("-", "").replace("/", "")
    variants = {str(date), digits}
    if len(digits) == 8:
        y, m, d = digits[:4], digits[4:6], digits[6:8]
        variants.update({f"{y}-{m}-{d}", f"{y}/{m}/{d}"})
    return list(variants)


def rows_for_date(conn, date):
    """All flightline rows flown on ``date`` (tolerant of date formatting).

    Returns a list of ``sqlite3.Row`` ordered by ``folder_name``.
    """
    variants = _date_variants(date)
    placeholders = ", ".join("?" for _ in variants)
    cur = conn.execute(
        f"SELECT * FROM {TABLE} WHERE flight_date IN ({placeholders}) "
        f"ORDER BY folder_name",
        variants,
    )
    return cur.fetchall()


def lookup_flightline(conn, flightline_name, date=None):
    """Return the row for ``flightline_name`` (optionally constrained to ``date``).

    Matching is case-insensitive on ``flightline_name``. Raises ``LookupError``
    with the available names if there is no unique match.
    """
    sql = f"SELECT * FROM {TABLE} WHERE lower(flightline_name) = lower(?)"
    params = [flightline_name]
    if date is not None:
        variants = _date_variants(date)
        sql += " AND flight_date IN (" + ", ".join("?" for _ in variants) + ")"
        params += variants
    rows = conn.execute(sql, params).fetchall()

    if not rows:
        available = [r["flightline_name"] for r in conn.execute(
            f"SELECT flightline_name FROM {TABLE} ORDER BY folder_name")]
        raise LookupError(
            f"No flightline named {flightline_name!r}"
            + (f" on {date}" if date else "")
            + f". Available: {available}")
    if len(rows) > 1:
        folders = [r["folder_name"] for r in rows]
        raise LookupError(
            f"Ambiguous flightline {flightline_name!r}: matches folders {folders}. "
            f"Pass a date to disambiguate.")
    return rows[0]


def rx_window_of(row):
    """Return the ``rx_window`` string for a row, or ``None`` if unset/absent.

    The value is the DAQ receive window as entered in the DB (e.g. ``"21.5us"``,
    ``"24us"``, ``"0"``). ``None`` means the column is missing/blank, which the
    driver treats as the nominal window for the region.
    """
    try:
        val = row["rx_window"]
    except (IndexError, KeyError):
        return None
    if val is None:
        return None
    val = str(val).strip()
    return val if val != "" else None
