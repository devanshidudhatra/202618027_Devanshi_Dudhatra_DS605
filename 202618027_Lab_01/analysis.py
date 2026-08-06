import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Read cleaned data
df = pd.read_csv("cleaned_books.csv")

# -------------------------------
# Summary Statistics
# -------------------------------

print("\nSummary Statistics\n")
print(df.describe())

print("\nBooks in each category\n")
print(df["category"].value_counts())

print("\nAverage price by category\n")
print(df.groupby("category")["price"].mean())

print("\nMissing Values\n")
print(df.isnull().sum())

# -------------------------------
# Plot 1 : Price Distribution
# -------------------------------

plt.figure(figsize=(8,5))

plt.hist(df["price"], bins=20)

plt.title("Price Distribution")
plt.xlabel("Price (£)")
plt.ylabel("Number of Books")

plt.savefig("price_distribution.png")
plt.show()

# -------------------------------
# Plot 2 : Rating Distribution
# -------------------------------

plt.figure(figsize=(6,5))

df["rating"].value_counts().sort_index().plot(kind="bar")

plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Number of Books")

plt.savefig("rating_distribution.png")
plt.show()

# -------------------------------
# Plot 3 : Average Price by Category
# -------------------------------

avg_price = df.groupby("category")["price"].mean().sort_values()

plt.figure(figsize=(12,6))

avg_price.plot(kind="bar")

plt.title("Average Price by Category")
plt.xlabel("Category")
plt.ylabel("Average Price (£)")

plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig("average_price_category.png")
plt.show()

# -------------------------------
# Plot 4 : Price vs Rating
# -------------------------------

plt.figure(figsize=(7,5))

plt.scatter(df["rating"], df["price"])

plt.title("Price vs Rating")
plt.xlabel("Rating")
plt.ylabel("Price (£)")

plt.savefig("price_vs_rating.png")
plt.show()

plt.figure(figsize=(7,5))

df.boxplot(column="price", by="rating")

plt.title("Price Distribution by Rating")
plt.suptitle("")        
plt.xlabel("Rating")
plt.ylabel("Price (£)")
plt.savefig("Price Distribution by Rating")
plt.show()

# -------------------------------
# Plot 5 : Category vs Stock
# -------------------------------

stock = df.groupby("category")["stock"].mean().sort_values()

plt.figure(figsize=(12,6))

stock.plot(kind="bar")

plt.title("Average Stock by Category")
plt.xlabel("Category")
plt.ylabel("Average Stock")

plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig("category_stock.png")
plt.show()

# -------------------------------
# Word Cloud
# -------------------------------

text = " ".join(df["product_description"].astype(str))

wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color="white"
).generate(text)

plt.figure(figsize=(12,6))

plt.imshow(wordcloud)

plt.axis("off")

plt.title("Book Description Word Cloud")

plt.savefig("wordcloud.png")

plt.show()

# -------------------------------
# Highly Rated Books
# -------------------------------

print("\nTop Rated Books\n")

print(
    df[df["rating"] == 5][
        ["title", "category", "price"]
    ].head(10)
)

# -------------------------------
# Most Expensive Books
# -------------------------------

print("\nMost Expensive Books\n")

print(
    df.sort_values("price", ascending=False)[
        ["title", "category", "price"]
    ].head(10)
)

# -------------------------------
# Best Value Books
# -------------------------------

print("\nBest Value Books\n")

print(
    df.sort_values("value_score", ascending=False)[
        ["title", "price", "rating", "value_score"]
    ].head(10)
)

