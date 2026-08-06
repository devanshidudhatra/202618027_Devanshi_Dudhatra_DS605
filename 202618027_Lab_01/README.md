# Books to Scrape Data Pipeline using Scrapy

## Overview

This project demonstrates a complete data pipeline for collecting, preprocessing, analyzing, and visualizing book data from the **Books to Scrape** website using **Python** and **Scrapy**.

The pipeline consists of four major stages:

1. Data Scraping
2. Data Preprocessing
3. Data Visualization and Analysis
4. Insights and Interpretation

The project satisfies the requirements of the assignment by scraping book information, cleaning and transforming the data, generating meaningful visualizations, and deriving data-driven insights.

---

## Website

https://books.toscrape.com/

---

## Objectives

- Scrape book information from at least five catalogue pages.
- Extract all required attributes for each book.
- Store the scraped data in CSV format.
- Clean and preprocess the collected dataset.
- Create additional useful features.
- Generate visualizations for exploratory data analysis.
- Identify meaningful patterns and insights from the data.

---

## Dataset

The project scrapes information for **1000 books** available on the website.

The following attributes are collected:

- Title
- Category
- Price
- Rating
- Availability
- Product Description
- UPC
- Number of Reviews
- Product URL

---

## Project Structure

```
bookscrapper/
│
├── tutorial/
│   ├── spiders/
│   │   └── books.py
│   ├── settings.py
│   └── ...
│
├── raw_books.csv
├── cleaned_books.csv
├── report_task1.py
├── preprocessing.py
├── analysis.py
├── requirements.txt
└── README.md
```

---

## Technologies Used

- Python
- Scrapy
- Pandas
- Matplotlib
- WordCloud

---

## Task 1 – Data Scraping

The Scrapy spider:

- Starts from the catalogue page.
- Traverses multiple catalogue pages.
- Visits each individual book page.
- Extracts the required information.
- Stores the scraped records in `raw_books.csv`.

A separate script (`report_task1.py`) reports:

- Total records scraped
- Missing values
- Duplicate UPC values

---

## Task 2 – Data Preprocessing

The preprocessing stage performs the following operations:

- Removes unnecessary spaces.
- Handles missing descriptions.
- Removes duplicate books using UPC.
- Converts price into numeric format.
- Converts ratings from text to integers.
- Extracts the available stock count.

Additional features created:

- description_word_count
- price_band
- affordability_score
- value_score
- recommended

The cleaned dataset is saved as:

```
cleaned_books.csv
```

---

## Task 3 – Data Visualization and Analysis

The following visualizations are generated:

- Price Distribution
- Rating Distribution
- Category Distribution
- Average Price by Category
- Price vs Rating
- Word Cloud using Book Descriptions

The analysis also includes:

- Summary statistics
- Category-wise analysis
- Highly rated books
- Best value books
- Missing value analysis

---

## Key Insights

- The dataset contains 1000 books with unique UPC values.
- Only a small number of book descriptions are missing.
- Book prices range approximately from £10 to £60.
- No strong relationship is observed between book price and rating.
- Some categories contain significantly more books than others.
- Average book prices vary across different categories.
- Books with higher value scores provide better price-to-rating combinations.

---

## Limitations

- The dataset is collected from a single practice website.
- It represents only one point in time.
- The dataset does not include author information, publication year, or customer reviews.
- Ratings are limited to five discrete values.
- Results may not generalize to other online bookstores.

---

## How to Run

### 1. Clone the repository

```bash
git clone <repository-link>
cd bookscrapper
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Scrapy spider

```bash
scrapy crawl books
```

This generates:

```
raw_books.csv
```

### 4. Generate Task 1 report

```bash
python report_task1.py
```

### 5. Preprocess the dataset

```bash
python preprocessing.py
```

This generates:

```
cleaned_books.csv
```

### 6. Perform analysis

```bash
python analysis.py
```

This generates the required visualizations and summary statistics.

---

## Output Files

- raw_books.csv
- cleaned_books.csv
- price_distribution.png
- rating_distribution.png
- category_distribution.png
- average_price_by_category.png
- price_vs_rating.png
- wordcloud.png

---

## Author

**Devanshi Dudhatra**

MSc  Data Science

Dhirubhai Ambani University