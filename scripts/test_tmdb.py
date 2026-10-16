
import os

import requests
from dotenv import load_dotenv

load_dotenv()

token = os.environ["TMDB_READ_ACCESS_TOKEN"]

url = "https://api.themoviedb.org/3/search/movie"

response = requests.get(
    url,
    headers={
        "Authorization": f"Bearer {token}",
        "accept": "application/json",
    },
    params={
        "query": "Interstellar",
        "language": "en-US",
    },
    timeout=15,
)

response.raise_for_status()

movies = response.json()["results"]

for movie in movies[:5]:
    print(
        movie["id"],
        movie["title"],
        movie.get("release_date", "Unknown"),
    )
