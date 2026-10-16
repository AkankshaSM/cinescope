
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

API_URL = "https://en.wikipedia.org/w/api.php"
OUTPUT_DIR = Path("data/documents")

ARTICLES = {
    "interstellar": "Interstellar (film)",
    "arrival": "Arrival (film)",
    "ex_machina": "Ex Machina (film)",
}

HEADERS = {
    "User-Agent": "CineScope/0.1 (academic research project; https://github.com/AkankshaSM/cinescope)",
    "Accept": "application/json",
}


def fetch_article(title):
    response = requests.get(
        API_URL,
        headers=HEADERS,
        params={
            "action": "query",
            "prop": "extracts|info",
            "inprop": "url",
            "explaintext": 1,
            "redirects": 1,
            "titles": title,
            "format": "json",
            "formatversion": 2,
        },
        timeout=30,
    )
    response.raise_for_status()

    pages = response.json()["query"]["pages"]
    page = pages[0]

    if page.get("missing"):
        raise ValueError(f"Article not found: {title}")

    return {
        "title": page["title"],
        "url": page["fullurl"],
        "text": page.get("extract", ""),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "license": "CC BY-SA 4.0",
        "source": "Wikipedia",
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for filename, title in ARTICLES.items():
        try:
            article = fetch_article(title)

            if not article["text"].strip():
                raise ValueError("Article returned empty text")

            output_path = OUTPUT_DIR / f"{filename}.json"

            with output_path.open("w", encoding="utf-8") as file:
                json.dump(article, file, indent=2, ensure_ascii=False)

            print(
                f"Saved {article['title']} "
                f"({len(article['text'])} characters)"
            )

        except (requests.RequestException, ValueError, KeyError) as error:
            print(f"Failed {title}: {error}")

        time.sleep(1)


if __name__ == "__main__":
    main()
