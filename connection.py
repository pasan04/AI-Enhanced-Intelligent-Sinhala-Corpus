import psycopg2

conn = psycopg2.connect(
    dbname="sinhala_corpus",
    user="corpus_user",
    password="root",
    host="localhost",
    port="5432"
)
print("Connected successfully")
conn.close()
