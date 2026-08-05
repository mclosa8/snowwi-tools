"""Comment-preserving editing of MIRSL/SNOWWI ``.cfg`` files.

The processor config files (``azmcomp_template.cfg``, ``preprocess_*.cfg``,
...) are read by ``configobj``/``configparser`` downstream, but they carry
inline ``#`` comments and commented-out alternative values that we want to keep
intact. A line-based regex edit preserves those, unlike a full parse+rewrite.

Extracted from ``bin/grid_search_offsets.py`` so both that tool and
``bin/process_snowwi.py`` share one implementation.
"""

import re


def update_config_field(path, key, new_value):
    """Set ``key = new_value`` in a ``.cfg`` file, preserving comments.

    Only the value *before* any inline ``#`` comment is replaced; the comment
    is left untouched. The first matching (non-commented) occurrence of ``key``
    is updated.

    Returns ``0`` on success, ``-1`` if the key was not found or on error
    (kept as return codes rather than exceptions for backwards compatibility
    with the original grid-search caller).
    """
    try:
        pattern = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*)([^#]*)(.*)$")
        out_lines = []
        found = False

        with open(path, "r") as f:
            for line in f:
                m = pattern.match(line)
                if m:
                    # Replace only the value before any comment.
                    line = f"{m.group(1)}{new_value}{m.group(3)}\n"
                    found = True
                out_lines.append(line)

        if not found:
            return -1

        with open(path, "w") as f:
            f.writelines(out_lines)

        return 0

    except Exception:
        return -1


def get_config_field(path, key):
    """Return the active value of ``key`` in a ``.cfg`` file, or ``None``.

    Reads the first non-commented ``key = value`` line, stripping any inline
    ``#`` comment. Returns ``None`` if the key is absent or the file is
    unreadable (e.g. it does not exist yet).
    """
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=\s*([^#\n]*)")
    try:
        with open(path) as f:
            for line in f:
                m = pattern.match(line)
                if m:
                    return m.group(1).strip()
    except OSError:
        return None
    return None


def set_config_field(path, key, new_value):
    """Like :func:`update_config_field` but raises on failure.

    Convenience wrapper for callers that prefer an exception to a ``-1`` return
    code. Raises ``KeyError`` if ``key`` is not present in ``path``.
    """
    if update_config_field(path, key, new_value) != 0:
        raise KeyError(f"Field {key!r} not found (or unwritable) in {path}")
