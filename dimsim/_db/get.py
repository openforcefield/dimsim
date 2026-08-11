import sqlite3


def get_record(db_path: str, record_id: str) -> tuple[str, float]:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT filename, energy FROM records WHERE id = ?", (record_id,))
    row = cursor.fetchone()

    conn.close()

    if row is None:
        raise KeyError(f"No record found with id={record_id!r}")

    filename, energy = row
    return filename, energy
