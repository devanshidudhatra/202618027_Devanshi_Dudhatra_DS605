import pandas as pd

# Read scraped data
df = pd.read_csv("raw_books.csv")

# Total books
print("Total Records:", len(df))

# Missing values
print("\nMissing Values")
print(df.isnull().sum())

# Duplicate UPC
print("\nDuplicate UPCs:", df["UPC"].duplicated().sum())