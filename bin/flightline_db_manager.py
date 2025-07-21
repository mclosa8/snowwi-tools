#!/usr/bin/env python3

"""
    SNOWWI flightline DB manager

    Author: Marc Closa Tarres (MCT)
    Date: 2025/07/18
    Version: v0

    Changelog:
        - v0: Initial version - Jul 18, 2025 - MCT
"""

import boto3
import csv
import glob
import os
import pprint
import sqlite3
import tempfile

from urllib.parse import urlparse


COLUMNS = [
    ("flightline_name", "TEXT NOT NULL"),
    ("folder_name", "TEXT NOT NULL UNIQUE"),
    ("flight_date", "TEXT NOT NULL"),
    ("start_local_time", "TEXT NOT NULL"),
    ("start_end_time", "TEXT NOT NULL"),
    ("notes", "TEXT NOT NULL")
    # Add more fields here as needed
]


COLUMN_NAMES = [col[0] for col in COLUMNS]


WELCOME_MSG = """Welcome to the SNOWWI Flightline Database Manager.
"""


MENU = """
Choose an option:

    1) List all entries
    2) Add a new entry
    3) Delete an entry by ID
    4) Edit an existing entry
    5) Search entries
    6) Export to CSV
    7) Exit
"""

def connect_db(db_path):
    if is_s3_path(db_path):
        local_path, bucket, key = download_s3_to_tempfile(db_path)
        conn = sqlite3.connect(local_path)
        conn._original_s3_info = (local_path, bucket, key)  # stash for later upload
        return conn
    else:
        return sqlite3.connect(db_path)


def init_db(conn):
    col_defs = ",\n            ".join([f"{name} {ctype}" for name, ctype in COLUMNS])
    with conn:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS flightlines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                {col_defs}
            )
        """)


def recreate_table_ordered_by_folder(conn):
    col_defs = ",\n                ".join([f"{name} {ctype}" for name, ctype in COLUMNS])
    cols_csv = ", ".join(COLUMN_NAMES)

    with conn:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS flightlines_temp (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                {col_defs}
            )
        """)
        conn.execute(f"""
            INSERT INTO flightlines_temp ({cols_csv})
            SELECT {cols_csv} FROM flightlines
            ORDER BY folder_name
        """)
        conn.execute("DROP TABLE flightlines")
        conn.execute("ALTER TABLE flightlines_temp RENAME TO flightlines")


def list_entries(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM flightlines ORDER BY folder_name")
    rows = cursor.fetchall()
    print("\nCurrent Flightlines (ordered by folder_name):\n")
    print_table(rows)


def add_entry(conn):
    print("\nAdd a New Entry (type 'back' to return to main menu)\n")
    values = []
    for col in COLUMN_NAMES:
        val = input(f"{col.replace('_', ' ').capitalize()}: ")
        if val.lower() == 'back':
            return
        values.append(val)

    placeholders = ", ".join(["?" for _ in COLUMN_NAMES])
    cols_csv = ", ".join(COLUMN_NAMES)

    try:
        with conn:
            conn.execute(f"""
                INSERT INTO flightlines ({cols_csv})
                VALUES ({placeholders})
            """, values)
        print("Entry added successfully.")
        recreate_table_ordered_by_folder(conn)
        print("IDs reindexed in chronological order.\n")
    except sqlite3.IntegrityError:
        print("Entry with this folder_name already exists. Skipped.")


def delete_entry(conn):
    print("\nDelete an Entry (type 'back' to return to main menu)\n")
    while True:
        entry_id = input("Enter ID of the entry to delete: ")
        if entry_id.lower() == 'back':
            return
        if not entry_id.isdigit():
            print("Please enter a valid numeric ID or 'back'.")
            continue
        with conn:
            conn.execute("DELETE FROM flightlines WHERE id = ?", (entry_id,))
        recreate_table_ordered_by_folder(conn)
        print(f"Entry {entry_id} deleted and IDs reindexed.\n")
        break


def edit_entry(conn):
    print("\nEdit an Entry (type 'back' to return to main menu)\n")
    entry_id = input("Enter ID of the entry to edit: ")
    if entry_id.lower() == 'back': return
    if not entry_id.isdigit():
        print("Invalid ID.")
        return

    cursor = conn.cursor()
    cursor.execute("SELECT * FROM flightlines WHERE id = ?", (entry_id,))
    row = cursor.fetchone()
    if not row:
        print("Entry not found.")
        return

    current_values = dict(zip(["id"] + COLUMN_NAMES, row))
    new_values = []

    print("Leave fields blank to keep current value.\n")
    for col in COLUMN_NAMES:
        val = input(f"{col.replace('_', ' ').capitalize()} [{current_values[col]}]: ") or current_values[col]
        new_values.append(val)

    cols_set = ", ".join([f"{col} = ?" for col in COLUMN_NAMES])
    with conn:
        try:
            conn.execute(f"""
                UPDATE flightlines SET {cols_set}
                WHERE id = ?
            """, (*new_values, entry_id))
            recreate_table_ordered_by_folder(conn)
            print("Entry updated and IDs reindexed.\n")
        except sqlite3.IntegrityError:
            print("Folder name must be unique. Update failed.\n")


def search_entries(conn):
    print("\nSearch Entries (by date, flightline name, or keyword in notes)")
    keyword = input("Enter date (YYYY-MM-DD), flightline name, or keyword: ").strip()

    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM flightlines
        WHERE flight_date LIKE ?
           OR flightline_name LIKE ?
           OR notes LIKE ?
    """, (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"))
    rows = cursor.fetchall()

    print("\nMatching Entries:\n")
    print_table(rows)


def export_to_csv(conn):
    path = input("Enter filename to export to (e.g., output.csv): ").strip()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM flightlines ORDER BY folder_name")
    rows = cursor.fetchall()
    headers = [desc[0] for desc in cursor.description]

    try:
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        print(f"Exported to {path}.\n")
    except Exception as e:
        print(f"Failed to export: {e}")

def print_table(rows):
    if not rows:
        print("No entries found.\n")
        return

    headers = ["id"] + COLUMN_NAMES
    data = [headers] + [list(map(str, row)) for row in rows]

    # Compute max width per column
    col_widths = [max(len(str(item)) for item in col) for col in zip(*data)]

    def format_row(row):
        return " | ".join(item.ljust(width) for item, width in zip(row, col_widths))

    print(format_row(headers))
    print("-" * (sum(col_widths) + 3 * (len(col_widths) - 1)))

    for row in data[1:]:
        print(format_row(row))


def is_s3_path(path):
    return path.startswith("s3://")


def parse_s3_path(s3_path):
    parsed = urlparse(s3_path)
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')
    return bucket, key


def download_s3_to_tempfile(s3_path):
    bucket, key = parse_s3_path(s3_path)
    s3 = boto3.client("s3")

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    s3.download_fileobj(bucket, key, temp_file)
    temp_file.flush()
    return temp_file.name, bucket, key


def upload_tempfile_to_s3(temp_path, bucket, key):
    s3 = boto3.client("s3")
    with open(temp_path, "rb") as f:
        s3.upload_fileobj(f, bucket, key)


def main():
    print(WELCOME_MSG)
    
    dbs_in_curr_dir = glob.glob('*.db')

    print("Databases in CWD:")
    pprint.PrettyPrinter().pprint(dbs_in_curr_dir)

    db_path = input('Input flightline database path: ').strip()
    db_path = os.path.abspath(db_path)

    conn = connect_db(db_path)
    init_db(conn)

    while True:
        print(MENU)
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            list_entries(conn)
        elif choice == "2":
            add_entry(conn)
        elif choice == "3":
            delete_entry(conn)
        elif choice == "4":
            edit_entry(conn)
        elif choice == "5":
            search_entries(conn)
        elif choice == "6":
            export_to_csv(conn)
        elif choice == "7":
            print("Exiting. Goodbye.")
            # If this was an S3-backed DB, upload the modified local file
            if hasattr(conn, "_original_s3_info"):
                local_path, bucket, key = conn._original_s3_info
                conn.close()
                upload_tempfile_to_s3(local_path, bucket, key)
                os.remove(local_path)
                print(f"\nChanges saved and uploaded to s3://{bucket}/{key}")
            else:
                conn.close()
            break
        else:
            print("Invalid option. Try again.")
    



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted. Exiting...")
