
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env", override=True)

BASE_URL = "https://api.themoviedb.org/3"
OUTPUT_PATH = ROOT_DIR / "data/raw/movies.json"

TARGET_MOVIES = 75

MOVIE_IDS = [
    157336,
    27205,
    329865,
    286217,
    438631,
    1124,
    264660,
    335984,
    11324,
    37165,
]


def create_session(token):
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    })

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


def fetch_movie(movie_id, session):
    response = session.get(
        f"{BASE_URL}/movie/{movie_id}",
        params={
            "language": "en-US",
            "append_to_response": "credits,external_ids",
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def discover_movie_ids(session, pages=5):
    movie_ids = []

    for page in range(1, pages + 1):
        response = session.get(
            f"{BASE_URL}/discover/movie",
            params={
                "language": "en-US",
                "sort_by": "vote_count.desc",
                "vote_average.gte": 7,
                "vote_count.gte": 1000,
                "include_adult": "false",
                "page": page,
            },
            timeout=30,
        )
        response.raise_for_status()

        movie_ids.extend(
            movie["id"]
            for movie in response.json().get("results", [])
        )

        time.sleep(0.2)

    return movie_ids


def extract_movie(data):
    directors = [
        person["name"]
        for person in data.get("credits", {}).get("crew", [])
        if person.get("job") == "Director"
    ]

    return {
        "id": data["id"],
        "title": data["title"],
        "release_date": data.get("release_date"),
        "overview": data.get("overview"),
        "genres": [
            genre["name"]
            for genre in data.get("genres", [])
        ],
        "director": directors,
        "runtime": data.get("runtime"),
        "vote_average": data.get("vote_average"),
        "vote_count": data.get("vote_count"),
        "wikidata_id": (
            data.get("external_ids") or {}
        ).get("wikidata_id"),
    }


def main():
    token = os.environ["TMDB_READ_ACCESS_TOKEN"]
    session = create_session(token)

    existing = {}

    if OUTPUT_PATH.exists():
        existing_movies = json.loads(
            OUTPUT_PATH.read_text(encoding="utf-8")
        )

        existing = {
            movie["id"]: movie
            for movie in existing_movies
        }

    print(f"Existing movies: {len(existing)}")

    discovered = discover_movie_ids(session)

    movie_ids = list(dict.fromkeys(
        MOVIE_IDS + discovered
    ))

    movies = {}

    for movie_id in movie_ids:
        if len(movies) >= TARGET_MOVIES:
            break

        if movie_id in existing and existing[movie_id].get(
            "wikidata_id"
        ):
            movies[movie_id] = existing[movie_id]
            continue

        try:
            data = fetch_movie(movie_id, session)
            movie = extract_movie(data)

            if not movie["release_date"]:
                continue

            if not movie["overview"]:
                continue

            movies[movie_id] = movie

            print(
                f"[{len(movies)}/{TARGET_MOVIES}] "
                f"{movie['title']}"
            )

        except requests.RequestException as error:
            print(f"Failed movie {movie_id}: {error}")

        time.sleep(0.2)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            list(movies.values()),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"\nSaved {len(movies)} movies "
        f"to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
