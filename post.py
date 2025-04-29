import pandas as pd
import psycopg2
from sqlalchemy import create_engine

# Пример DataFrame
df = pd.read_csv(r"C:\Users\akylb\Downloads\IMDB Dataset.csv\IMDB Dataset.csv")
conn = psycopg2.connect(
    host="localhost",    
    database="ML", 
    user="postgres", 
    password="1234",
    port="5432"
)

engine = create_engine('postgresql+psycopg2://postgres:1234@localhost:5432/ML')

df.to_sql('ml_project', engine, if_exists='replace', index=False)

conn.close()
