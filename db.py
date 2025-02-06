import sqlite3

def create_temp_db():
    # Connect to an in-memory database
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE user_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT
        )
    ''')

    return conn, cursor

def add_message(conn, cursor, message):
    cursor.execute('''
        INSERT INTO user_data (message)
        VALUES (?)
    ''', (message,))

    conn.commit()

def get_messages(cursor):
    cursor.execute('SELECT * FROM user_data')
    data = cursor.fetchall()
    return data
