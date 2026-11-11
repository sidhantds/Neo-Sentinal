"""Live close approaches from NASA's NeoWs feed API.

The feed returns every close approach between two dates (at most 7 days per
request). parse_feed() flattens it into the same columns as the training
CSV, so live objects can be scored with no extra conversion.

Get a free key at https://api.nasa.gov. DEMO_KEY works but is rate limited
to about 30 requests per hour per IP address.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from datetime import date, timedelta

import pandas as pd

FEED_URL = "https://api.nasa.gov/neo/rest/v1/feed"
MAX_DAYS = 7


def api_key() -> str:
    return os.environ.get("NASA_API_KEY", "DEMO_KEY")


def parse_feed(payload: dict) -> pd.DataFrame:
    rows = []
    for approach_date, objects in payload.get("near_earth_objects", {}).items():
        for obj in objects:
            diameter = obj.get("estimated_diameter", {}).get("kilometers", {})
            for approach in obj.get("close_approach_data", []):
                if approach.get("orbiting_body", "Earth") != "Earth":
                    continue
                rows.append(
                    {
                        "id": obj.get("id"),
                        "name": obj.get("name"),
                        "close_approach_date": approach.get("close_approach_date", approach_date),
                        "est_diameter_min": diameter.get("estimated_diameter_min"),
                        "est_diameter_max": diameter.get("estimated_diameter_max"),
                        "relative_velocity": float(approach["relative_velocity"]["kilometers_per_hour"]),
                        "miss_distance": float(approach["miss_distance"]["kilometers"]),
                        "absolute_magnitude": obj.get("absolute_magnitude_h"),
                        "nasa_hazardous": bool(obj.get("is_potentially_hazardous_asteroid")),
                        "sentry_object": bool(obj.get("is_sentry_object")),
                        "nasa_url": obj.get("nasa_jpl_url"),
                    }
                )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["close_approach_date", "miss_distance"]).reset_index(drop=True)
    return df


def fetch_feed(start: date, end: date | None = None, key: str | None = None, timeout: int = 30) -> pd.DataFrame:
    end = end or start + timedelta(days=MAX_DAYS - 1)
    if end < start:
        raise ValueError("end date is before start date")
    if (end - start).days >= MAX_DAYS:
        raise ValueError(f"NeoWs allows at most {MAX_DAYS} days per request")
    query = urllib.parse.urlencode(
        {"start_date": start.isoformat(), "end_date": end.isoformat(), "api_key": key or api_key()}
    )
    req = urllib.request.Request(f"{FEED_URL}?{query}", headers={"User-Agent": "neo-sentinel"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return parse_feed(json.load(resp))
