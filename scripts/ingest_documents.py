
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT_DIR = Path(__file__).resolve().parent.parent
MOVIES_PATH = ROOT_DIR / "data/raw/movies.json"
OUTPUT_DIR = ROOT_DIR / "data/documents"

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"

HEADERS = {
    "User-Agent": (
        "CineScope/0.2 "
        "(academic research project; "
        "https://github.com/AkankshaSM/cinescope)"
    ),
    "Accept": "application/json",
}


def create_session():
    session = requests.Session()
    session.headers.update(HEADERS)

    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        respect_retry_after_header=True,
    )

    session.mount(
        "https://",
        HTTPAdapter(max_retries=retries),
    )

    return session


def resolve_wikipedia_title(wikidata_id, session):
    response = session.get(
        WIKIDATA_API,
        params={
            "action": "wbgetentities",
            "ids": wikidata_id,
            "props": "sitelinks",
            "format": "json",
        },
        timeout=30,
    )
    response.raise_for_status()

    entity = response.json().get(
        "entities", {}
    ).get(wikidata_id, {})

    return (
        entity.get("sitelinks", {})
        .get("enwiki", {})
        .get("title")
    )


def fetch_article(title, session):
    response = session.get(
        WIKIPEDIA_API,
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
        raise ValueError(
            f"Wikipedia article missing: {title}"
        )

    return {
        "title": page["title"],
        "url": page["fullurl"],
        "text": page.get("extract", ""),
        "retrieved_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "license": "CC BY-SA 4.0",
        "source": "Wikipedia",
    }


def main():
    movies = json.loads(
        MOVIES_PATH.read_text(encoding="utf-8")
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    session = create_session()

    saved = 0
    skipped = 0

    for movie in movies:
        movie_id = movie["id"]
        wikidata_id = movie.get("wikidata_id")

        output_path = (
            OUTPUT_DIR / f"{movie_id}.json"
        )

        if output_path.exists():
            skipped += 1
            continue

        if not wikidata_id:
            print(
                f"Skipping {movie['title']}: "
                "No Wikidata ID"
            )
            skipped += 1
            continue

        try:
            title = resolve_wikipedia_title(
                wikidata_id,
                session,
            )

            if not title:
                raise ValueError(
                    "No English Wikipedia article"
                )

            article = fetch_article(
                title,
                session,
            )

            if len(article["text"].strip()) < 500:
                raise ValueError(
                    "Article text too short"
                )

            article["tmdb_id"] = movie_id
            article["movie_title"] = movie["title"]
            article["wikidata_id"] = wikidata_id

            output_path.write_text(
                json.dumps(
                    article,
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            saved += 1

            print(
                f"Saved {movie['title']} "
                f"({len(article['text'])} characters)"
            )

        except (
            requests.RequestException,
            ValueError,
            KeyError,
        ) as error:
            print(
                f"Failed {movie['title']}: {error}"
            )
            skipped += 1

        time.sleep(1)

    print(
        f"\nNew documents: {saved}"
        f"\nSkipped: {skipped}"
    )


if __name__ == "__main__":
    main()
