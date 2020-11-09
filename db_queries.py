import sqlite3 as sql

# start connection to database
conn = sql.connect('automl.db')
c = conn.cursor()


def get_users():
    
    c.execute('SELECT * FROM users')
    return {u[1]:u[2] for u in c.fetchall()}
 