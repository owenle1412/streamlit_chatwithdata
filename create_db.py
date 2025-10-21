import pandas as pd
from sqlalchemy import create_engine
import os

DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "sales_qtr.csv")  # replace with your file
SQLITE_PATH = os.path.join(DATA_DIR, "sales.db")

def csv_to_sqlite(csv_path=CSV_PATH, sqlite_path=SQLITE_PATH, table_name="sales"):
    df = pd.read_csv(csv_path)
    engine = create_engine(f"sqlite:///{sqlite_path}", echo=False)
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    print(f"Wrote {len(df)} rows to {sqlite_path} table '{table_name}'")

if __name__ == "__main__":
    csv_to_sqlite()