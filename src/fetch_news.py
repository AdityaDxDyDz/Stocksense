from newsapi import NewsApiClient
import requests
import os
from dotenv import load_dotenv

load_dotenv()

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
ALPHA_KEY = os.getenv("ALPHA_KEY")

newsapi = NewsApiClient(api_key=NEWSAPI_KEY)


def fetch_news(company):

    try:

        # detect NSE stock
        if "_NS" in company or ".NS" in company:

            symbol = company.replace("_NS", ".NS")

            url = "https://www.alphavantage.co/query"

            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": symbol,
                "apikey": ALPHA_KEY
            }

            response = requests.get(url, params=params).json()

            headlines = []

            if "feed" in response:

                keyword = symbol.replace(".NS", "").upper()

                for article in response["feed"]:

                    title = article.get("title")

                    if title and keyword in title.upper():
                        headlines.append(title)

                    if len(headlines) == 5:
                        break

            headlines = list(set(headlines))

            print("Fetched NSE Headlines:", headlines)

            return headlines

        # S&P500 stocks (existing logic unchanged)
        else:

            company = company.replace("_NS", "").replace(".NS", "")

            headlines = []

            queries = [
                company,
                f"{company} stock",
                f"{company} company",
                f"{company} business"
            ]

            for q in queries:

                articles = newsapi.get_everything(
                    q=q,
                    language="en",
                    sort_by="publishedAt",
                    page_size=5
                )

                for article in articles["articles"]:

                    title = article.get("title")

                    if title:
                        headlines.append(title)

                if headlines:
                    break

            headlines = list(set(headlines))

            print("Fetched US Headlines:", headlines)

            return headlines

    except Exception as e:

        print("❌ Error fetching news:", e)

        return []