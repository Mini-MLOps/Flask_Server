import pymysql


class DB_connect:
    def __init__(self):
        self.conn = pymysql.connect(
            host="211.62.99.58",
            port=3326,
            user="root",
            password="1234",
            db="mlops_db",
            charset="utf8",
        )
        self.conn.ping(reconnect=True)
        self.curs = self.conn.cursor()

    def select(self, sql):
        self.curs.execute(sql)
        data = self.curs.fetchall()
        return data

    def insert(self, sql, data):
        self.curs.execute(sql, data)
        self.conn.commit()

    def truncate(self, table_name):
        truncate_query = f"TRUNCATE {table_name};"
        self.curs.execute(truncate_query)
        self.conn.commit()

    def close(self):
        self.curs.close()
        self.conn.close()
