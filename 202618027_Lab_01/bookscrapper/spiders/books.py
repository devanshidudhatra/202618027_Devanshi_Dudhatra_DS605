import scrapy


# Spider class
class BooksSpider(scrapy.Spider):

    # Spider name
    name = "books"

    # Starting page
    start_urls = [
        "https://books.toscrape.com/catalogue/page-1.html"
    ]

    # This method extracts all book links from each catalogue page
    def parse(self, response):

        # Get all book links on the current page
        books = response.css("article.product_pod h3 a::attr(href)").getall()

        # Visit every book page
        for book in books:
            yield response.follow(book, callback=self.parse_book)

        # Find the next page
        next_page = response.css("li.next a::attr(href)").get()

        # Continue until no next page exists
        if next_page:
            yield response.follow(next_page, callback=self.parse)

    # This method extracts data from one book page
    def parse_book(self, response):

        # Dictionary to store table information
        product_info = {}

        # Read every row from the product information table
        for row in response.css("table tr"):

            key = row.css("th::text").get()
            value = row.css("td::text").get()

            product_info[key] = value

        # Product description
        description = response.css("#product_description + p::text").get()

        if description is None:
            description = ""

        # Rating (One, Two, Three, Four, Five)
        rating = response.css("p.star-rating").attrib["class"].split()[-1]

        # Category
        category = response.css(
            "ul.breadcrumb li:nth-child(3) a::text"
        ).get()

        # Availability
        availability = " ".join(
            response.css("p.availability::text").getall()
        ).strip()

        # Return one record
        yield {

            "title": response.css("h1::text").get(),

            "category": category,

            "price": response.css(
                "p.price_color::text"
            ).get(),

            "rating": rating,

            "availability": availability,

            "product_description": description,

            "UPC": product_info.get("UPC"),

            "number_of_reviews":
            product_info.get("Number of reviews"),

            "product_url": response.url

        }