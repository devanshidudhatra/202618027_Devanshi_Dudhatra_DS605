import pandas as pd
import numpy as np


df = pd.read_csv("raw_books.csv")

# 1. Remove extra spaces from text columns
cols = ["title","category","availability","product_description"]
for col in  cols:
    df[col] = df[col].fillna("").str.strip()

 
# 2. Remove duplicate books using UPC
df = df.drop_duplicates(subset="UPC")

 
# 3. Handle missing descriptions
df["product_description"] = df["product_description"].replace("", "No Description")

 
# 4. Convert price to numeric
df["price"] = df["price"].str.replace("£", "", regex=False).astype(float)

 
# 5. Convert ratings into numbers
rm = {"One": 1,"Two": 2,"Three": 3,"Four": 4,"Five": 5}
df["rating"] = df["rating"].map(rm)

 
# 6. Extract stock count
df["stock"] = df["availability"].str.extract(r"(\d+)").fillna(0).astype(int)

 
# 7. Feature Engineering
 
# Feature 1 : Number of words in description
df["description_word_count"] = df["product_description"].str.split().str.len()

# Feature 2 : Price Band
def pb(price):
    if price < 20:
        return "Cheap"
    elif price < 40:
        return "Medium"
    elif price < 60:
        return "Expensive"
    else:
        return "Luxury"

df["pb"] = df["price"].apply(pb)

# Feature 3 : Affordability Score
# Higher score means cheaper book
df["affordability_score"] = round(100 / df["price"], 2)

# Feature 4 : Value Score
# Higher rating and lower price gives higher value
df["value_score"] = round(df["rating"] / df["price"], 3)

# Feature 5 : Recommended Book
df["recommended"] = np.where((df["rating"] >= 4) & (df["price"] <= 35),"Yes","No")

 
df.to_csv("cleaned_books.csv", index=False)
print("Preprocessing Done.")