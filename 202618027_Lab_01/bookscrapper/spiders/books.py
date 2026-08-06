import scrapy


# Spider class
class BSpider(scrapy.Spider):

    # Spider name
    name = "books"

    # Starting page from where scarpping starts
    start_urls = [
        "https://books.toscrape.com/catalogue/page-1.html"
    ]

    # parse is a inbuilt function and in this code is written to get each book page link and it simply means parsing the webpage
    def parse(self, response):
        # this finds all the book links on the webpage
        books = response.css("article.product_pod h3 a::attr(href)").getall()
        for book in books:
            yield response.follow(book, callback=self.parse_book)
        # to get the next page
        next_page = response.css("li.next a::attr(href)").get()
        if next_page:
            yield response.follow(next_page, callback=self.parse)

    # this function is to parse each book link
    def parse_book(self, response):
        product_info = {}
        for row in response.css("table tr"):
            key = row.css("th::text").get()
            value = row.css("td::text").get()
            product_info[key] = value

        description = response.css("#product_description + p::text").get()
        if description is None:
            description = ""

        # Rating 
        rating = response.css("p.star-rating").attrib["class"].split()[-1]

        # Category
        category = response.css("ul.breadcrumb li:nth-child(3) a::text").get()

        # Availability
        availability = " ".join(response.css("p.availability::text").getall()).strip()

        # Return one record
        yield {
            "title": response.css("h1::text").get(),
            "category": category,
            "price": response.css("p.price_color::text").get(),
            "rating": rating,
            "availability": availability,
            "product_description": description,
            "UPC": product_info.get("UPC"),
            "number_of_reviews": product_info.get("Number of reviews"),
            "product_url": response.url
        }

# In the whole code why yield is used instead of return is because yield is used to return a generator object 
# which can be iterated over and it allows the function to produce a series of values over time, 
# rather than computing them all at once and sending them back. 
# This is particularly useful in web scraping where you may want to process each item as it is found, 
# rather than waiting for the entire page to be processed before returning results.
# in short return will collect all the values and return them at once but yield will return one value at a time and then continue from where it left off


# ------------------------------- UNDERSTANDING THE CODE -------------------------------

# 1. We first import scrapy because it provides all the tools required for web scraping.

# 2. We create a Spider class because Scrapy works using spiders.
#    A spider is like a robot that knows:
#    - where to start,
#    - what pages to visit,
#    - what information to collect.

# 3. name = "books"
#    Every spider must have a unique name.
#    When we run "scrapy crawl books", Scrapy searches for the spider
#    whose name is "books" and starts executing it.

# 4. start_urls
#    This tells the spider from which webpage it should begin scraping.
#    Here it starts from catalogue page 1.

# 5. parse()
#    parse() is an inbuilt callback function.
#    Whenever Scrapy downloads a webpage, it automatically sends that webpage
#    to parse() for processing.
#    In this project, parse() is responsible for:
#       - finding all book links on the current page.
#       - opening every individual book page.
#       - finding the next catalogue page.

# 6. response
#    response is the webpage downloaded by Scrapy.
#    It contains:
#       - HTML source
#       - URL
#       - status code
#       - headers
#    We use response.css() to extract information from that webpage.

# 7. response.css()
#    It searches the HTML using CSS selectors.
#    It works similar to selecting HTML elements using CSS in web development.

# 8. article.product_pod h3 a::attr(href)
#    This selector means:
#       article.product_pod -> each book card
#       h3 -> heading containing the book title
#       a -> hyperlink of the book
#       ::attr(href) -> extract the href attribute (book page link)

# 9. getall()
#    Returns all matching values as a list.
#    Here it returns links of all books present on one catalogue page.

# 10. for book in books
#     Loop through every book link one by one.

# 11. response.follow(book, callback=self.parse_book)
#     Open the selected book page.
#     After downloading that page,
#     automatically call parse_book().

# 12. next_page
#     Finds the "Next" button available at the bottom of the catalogue page.

# 13. if next_page
#     If another catalogue page exists,
#     visit that page and again call parse().
#     This continues until there are no more pages.

# ------------------------------- parse_book() -------------------------------

# 14. parse_book()
#     This function extracts information from one individual book page.

# 15. product_info = {}
#     Creates an empty dictionary to temporarily store values
#     from the product information table.

# 16. table tr
#     Selects every row of the information table.

# 17. th::text
#     Gets the table heading.
#     Example:
#     UPC
#     Product Type
#     Price (excl. tax)

# 18. td::text
#     Gets the corresponding value of that heading.

# 19. product_info[key] = value
#     Stores the information as key-value pairs.
#
#     Example:
#     {
#         "UPC": "12345",
#         "Number of reviews": "0"
#     }

# 20. Product Description
#     Some books do not have descriptions.
#     If description is missing,
#     an empty string is stored instead of producing an error.

# 21. Rating
#     The website stores ratings as CSS classes like:
#     <p class="star-rating Three">
#
#     split() separates the words:
#     ["star-rating", "Three"]
#
#     [-1] selects the last word:
#     "Three"

# 22. Category
#     The breadcrumb shows:
#     Home > Books > Travel > Book Name
#
#     nth-child(3) selects the third item,
#     which is the category.

# 23. Availability
#     Availability text comes in multiple pieces.
#     getall() returns all pieces.
#     " ".join() combines them into one sentence.
#     strip() removes extra spaces.

# ------------------------------- yield -------------------------------

# 24. yield returns one dictionary at a time.
#     Every dictionary represents one book.

#     Book 1 -> yield
#     Book 2 -> yield
#     Book 3 -> yield

#     This continues until all books are scraped.

#     Using yield saves memory because Scrapy processes one item
#     immediately instead of waiting for all books to finish.

#     If return were used, the function would stop after returning
#     the first dictionary, and no further books would be processed.

# ------------------------------- Overall Flow -------------------------------

# Start from Page 1
#        |
#        v
# parse()
#        |
# Find all book links
#        |
# Open each book page
#        |
# parse_book()
#        |
# Extract required information
#        |
# yield one dictionary
#        |
# Save into CSV
#        |
# Go to next catalogue page
#        |
# Repeat until there is no next page.

# Final Output:
# One row in the CSV file represents one book.
# The spider continues scraping until all catalogue pages have been visited.