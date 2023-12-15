import pymysql


class DB_connect:
    def __init__(self):
        self.connect()

    def connect(self):
        self.conn = pymysql.connect(
            host="211.62.99.58",
            port=3326,
            user="root",
            password="1234",
            db="mlops_db",
            charset="utf8",
        )
        self.curs = self.conn.cursor()

    def select(self, sql):
        self.connect()
        self.curs.execute(sql)
        data = self.curs.fetchall()
        self.close()
        return data

    def insert(self, sql, data):
        self.connect()
        self.curs.execute(sql, data)
        self.conn.commit()
        self.close()

    def truncate(self, table_name):
        self.connect()
        self.curs.execute(f"TRUNCATE {table_name};")
        self.conn.commit()
        self.close()

    def close(self):
        self.curs.close()
        self.conn.close()
