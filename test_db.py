import pymysql
from config import DB_CONFIG

try:
    connection = pymysql.connect(**DB_CONFIG)

    with connection.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM foods;")
        result = cursor.fetchone()
        print("Connection successful!")
        print("Foods count:", result[0])

    connection.close()

except Exception as e:
    print("Connection failed!")
    print(e)