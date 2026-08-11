import sqlite3


def insert_record(db_path: str, record_id: str, filename: str, energy: float):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("INSERT INTO records (id, filename, energy) VALUES (?, ?, ?)", (record_id, filename, energy))

    conn.commit()
    conn.close()
