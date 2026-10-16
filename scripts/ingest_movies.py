
import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.themoviedb.org/3"

MOVIE_IDS = [
    157336,  # Interstellar
    27205,   # Inception
    329865,  # Arrival
    286217,  # The Martian
    438631,  # Dune
    1124,    # The Prestige
    264660,  # Ex Machina
    335984,  # Blade Runner 2049
    11324,   # Shutter Island
    37165,   # The Truman Show
]


def fetch_movie(movie_id, token):
    response = requests.get(
        f"{BASE_URL}/movie/{movie_id}",
        headers={
            "Authorization": f"Bearer {token}",
            "accept": "application/json",
        },
        params={
            "language": "en-US",
            "append_to_response": "credits",
        },
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


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
    }


def main():
    token = os.environ["TMDB_READ_ACCESS_TOKEN"]
    movies = []

    for movie_id in MOVIE_IDS:
        try:
            data = fetch_movie(movie_id, token)
            movie = extract_movie(data)
            movies.append(movie)
            print(f"Fetched: {movie['title']}")
        except requests.RequestException as error:
            print(f"Failed movie {movie_id}: {error}")

    output_path = Path("data/raw/movies.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(movies, file, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(movies)} movies to {output_path}")


if __name__ == "__main__":
    main()
