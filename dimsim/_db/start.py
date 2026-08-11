import sqlite3


def create_database(db_path: str):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # not sure if we should be tracking file name (easy enough, brittle)
    # or storing the full 3-D coordinates in the database
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS records (
            id TEXT PRIMARY KEY CHECK(length(id) = 16),
            filename TEXT NOT NULL,
            energy REAL NOT NULL
        )
    """)

    conn.commit()
    conn.close()
