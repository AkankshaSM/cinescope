
import json
import sqlite3
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT_DIR / "data/raw/movies.json"
DATABASE_PATH = ROOT_DIR / "data/movies.db"


def main():
    movies = json.loads(
        INPUT_PATH.read_text(encoding="utf-8")
    )

    ids = [movie["id"] for movie in movies]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate TMDB movie IDs found")

    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("PRAGMA foreign_keys = ON")

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

        connection.execute("""
            CREATE TEMP TABLE current_movie_ids (
                id INTEGER PRIMARY KEY
            )
        """)

        connection.executemany(
            "INSERT INTO current_movie_ids (id) VALUES (?)",
            [(movie_id,) for movie_id in ids],
        )

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

        connection.execute("""
            DELETE FROM movie_genres
            WHERE movie_id NOT IN (
                SELECT id FROM current_movie_ids
            )
        """)

        connection.execute("""
            DELETE FROM movies
            WHERE id NOT IN (
                SELECT id FROM current_movie_ids
            )
        """)

        total_movies = connection.execute(
            "SELECT COUNT(*) FROM movies"
        ).fetchone()[0]

        total_genres = connection.execute(
            "SELECT COUNT(*) FROM movie_genres"
        ).fetchone()[0]

    print(f"Movies: {total_movies}")
    print(f"Movie-genre relationships: {total_genres}")
    print(f"Database: {DATABASE_PATH}")


if __name__ == "__main__":
    main()
