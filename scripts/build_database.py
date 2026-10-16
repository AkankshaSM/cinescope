
import json
import sqlite3
from pathlib import Path

INPUT_PATH = Path("data/raw/movies.json")
DATABASE_PATH = Path("data/movies.db")


def main():
    with INPUT_PATH.open("r", encoding="utf-8") as file:
        movies = json.load(file)

    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS movies (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                release_date TEXT,
                overview TEXT,
                director TEXT,
                runtime INTEGER,
                vote_average REAL,
                vote_count INTEGER
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS movie_genres (
                movie_id INTEGER NOT NULL,
                genre TEXT NOT NULL,
                PRIMARY KEY (movie_id, genre),
                FOREIGN KEY (movie_id) REFERENCES movies(id)
            )
        """)

        for movie in movies:
            connection.execute(
                """
                INSERT INTO movies (
                    id, title, release_date, overview,
                    director, runtime, vote_average, vote_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    release_date = excluded.release_date,
                    overview = excluded.overview,
                    director = excluded.director,
                    runtime = excluded.runtime,
                    vote_average = excluded.vote_average,
                    vote_count = excluded.vote_count
                """,
                (
                    movie["id"],
                    movie["title"],
                    movie.get("release_date"),
                    movie.get("overview"),
                    ", ".join(movie.get("director", [])),
                    movie.get("runtime"),
                    movie.get("vote_average"),
                    movie.get("vote_count"),
                ),
            )

            connection.execute(
                "DELETE FROM movie_genres WHERE movie_id = ?",
                (movie["id"],),
            )

            connection.executemany(
                """
                INSERT INTO movie_genres (movie_id, genre)
                VALUES (?, ?)
                """,
                [
                    (movie["id"], genre)
                    for genre in movie.get("genres", [])
                ],
            )

    print(f"Saved {len(movies)} movies to {DATABASE_PATH}")


if __name__ == "__main__":
    main()
