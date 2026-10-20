
import sqlite3
from pathlib import Path

DATABASE_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "movies.db"
)


def search_movies(
    genre: str | None = None,
    director: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    min_rating: float | None = None,
    limit: int = 10,
) -> list[dict]:
    """
    Search the local movie catalog using optional filters.

    Returns a list of matching movies sorted by TMDB rating.
    """

    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")

    if (
        year_from is not None
        and year_to is not None
        and year_from > year_to
    ):
        raise ValueError("year_from cannot exceed year_to")

    query = """
        SELECT
            m.id,
            m.title,
            m.release_date,
            m.director,
            m.runtime,
            m.vote_average,
            m.vote_count,
            GROUP_CONCAT(mg.genre, ', ') AS genres
        FROM movies AS m
        LEFT JOIN movie_genres AS mg
            ON m.id = mg.movie_id
        WHERE 1 = 1
    """

    parameters = []

    if genre:
        query += """
            AND EXISTS (
                SELECT 1
                FROM movie_genres AS g
                WHERE g.movie_id = m.id
                  AND LOWER(g.genre) = LOWER(?)
            )
        """
        parameters.append(genre)

    if director:
        query += " AND LOWER(m.director) LIKE LOWER(?)"
        parameters.append(f"%{director}%")

    if year_from is not None:
        query += " AND CAST(SUBSTR(m.release_date, 1, 4) AS INTEGER) >= ?"
        parameters.append(year_from)

    if year_to is not None:
        query += " AND CAST(SUBSTR(m.release_date, 1, 4) AS INTEGER) <= ?"
        parameters.append(year_to)

    if min_rating is not None:
        query += " AND m.vote_average >= ?"
        parameters.append(min_rating)

    query += """
        GROUP BY m.id
        ORDER BY m.vote_average DESC, m.vote_count DESC
        LIMIT ?
    """
    parameters.append(limit)

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row

        rows = connection.execute(
            query,
            parameters,
        ).fetchall()

    return [dict(row) for row in rows]


if __name__ == "__main__":
    import json

    movies = search_movies(
        genre="Science Fiction",
        year_from=2010,
        min_rating=7.0,
    )

    print(json.dumps(movies, indent=2))
