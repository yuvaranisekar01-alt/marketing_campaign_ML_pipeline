import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import sys
import os
from dotenv import load_dotenv

load_dotenv()

# 1. db connection
def get_connection():
    conn = psycopg2.connect(
        host     = os.getenv('DB_HOST'),
        port     = os.getenv('DB_PORT'),
        database = os.getenv('DB_NAME'),   
        user     = os.getenv('DB_USER'),      
        password = os.getenv('DB_PASSWORD')  
    )
    return conn


def get_engine():
    engine = create_engine(
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)
    return engine

# 2. create tables
def create_tables():
    conn   = get_connection()
    cursor = conn.cursor()

    with open('sql/queries.sql', 'r') as f:
        sql = f.read()

    
    create_statements = [s.strip() for s in sql.split(';')
                         if s.strip().upper().startswith('CREATE')]

    for statement in create_statements:
        cursor.execute(statement)

    conn.commit()
    cursor.close()
    conn.close()
    print("✅ Tables created!")

# 3. Insert raw data
def raw_data():
    engine = get_engine()

    # Load all 3 datasets
    df1 = pd.read_csv('data/raw/nykaa_campaign_data_with_nulls.csv', encoding='utf-8')
    df2 = pd.read_csv('data/raw/purplle_campaign_data_with_nulls.csv', encoding='utf-8')
    df3 = pd.read_csv('data/raw/tira_campaign_data_with_nulls.csv', encoding='utf-8')
    
    df1['Brand'] = 'Nykaa'
    df2['Brand'] = 'Purplle'
    df3['Brand'] = 'Tira'

    df = pd.concat([df1, df2, df3], ignore_index=True)

    df.columns = df.columns.str.lower()

    df.to_sql(
        'raw_campaigns',
        engine,
        if_exists='replace',   
        index=False,
        chunksize=1000 )
    
    print(f"✅ Raw data inserted! Rows: {len(df)}")


#  4. insert feature engineered data 
def insert_feature_data():
    engine = get_engine()

    df = pd.read_csv('data/processed/feature_engineered_df.csv')
    df.columns = df.columns.str.lower()

    df.to_sql(
        'feature_engineered_campaigns',
        engine,
        if_exists='replace',
        index=False,
        chunksize=1000 )
    
    print(f"✅ Feature engineered data inserted! Rows: {len(df)}")

#  5. insert predictions 
def insert_predictions(predictions_df):
    engine = get_engine()

    predictions_df.columns = predictions_df.columns.str.lower()
    predictions_df.to_sql(
        'model_predictions',
        engine,
        if_exists='append',    
        index=False,
        chunksize=1000 )
    print(f"✅ Predictions inserted! Rows: {len(predictions_df)}")

# 6. run a query and return dataframe 
def run_query(query):
    engine = get_engine()
    df = pd.read_sql(query, engine)
    return df


#  main 
if __name__ == "__main__":
    create_tables()
    raw_data()
    insert_feature_data()
    print("\n✅ Database setup complete!")
    print("Run sql/queries.sql in pgAdmin to verify data.")


