import MySQLdb

# 接続する
conn = MySQLdb.connect(
    user='root',
    passwd='root',
    host='localhost',
    port='3306',
    db='mysql'
)

print(conn.is_connected())