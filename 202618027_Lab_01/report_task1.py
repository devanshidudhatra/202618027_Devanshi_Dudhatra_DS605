import pandas as pd

df = pd.read_csv("raw_books.csv")
print("Total Records:", len(df))
print("\nMissing Values")
print(df.isnull().sum())
print("\nDuplicate UPCs:", df["UPC"].duplicated().sum())