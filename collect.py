"""Fetch public full-text posts with on-disk page caching and provenance."""
import argparse
import html
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from common import config, digest, write_jsonl


def clean_html(value):
    soup = BeautifulSoup(value or "", "html.parser")
    for tag in soup(["script", "style", "nav", "form", "button", "iframe"]):
        tag.decompose()
    # Preserve paragraphs rather than inserting a newline at every inline span.
    for tag in soup.find_all(["p", "div", "li", "h1", "h2", "h3", "h4", "blockquote", "br"]):
        tag.insert_before("\n\n")
        tag.insert_after("\n\n")
    return "\n\n".join(" ".join(p.split()) for p in soup.get_text().split("\n\n") if p.strip())


class Client:
    def __init__(self, cache):
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "rationalist-lora-personal-research/0.1"

    def get(self, url, params):
        key = digest(json.dumps([url, params], sort_keys=True))
        path = self.cache / (key + ".json")
        if path.exists():
            return json.loads(path.read_text())
        for attempt in range(4):
            try:
                response = self.session.get(url, params=params, timeout=60)
                response.raise_for_status()
                value = response.json()
                if isinstance(value, dict) and value.get("errors"):
                    raise ValueError(str(value["errors"]))
                path.write_text(json.dumps(value))
                time.sleep(0.7)
                return value
            except (requests.RequestException, ValueError):
                if attempt == 3:
                    raise
                time.sleep(2 ** (attempt + 1))


def fetch_forum(client, source, limit):
    for offset in range(0, limit, 50):
        count = min(50, limit - offset)
        query = """{posts(input:{terms:{view:"top",meta:null,limit:%d,offset:%d}}){results{
            _id title pageUrl postedAt baseScore af user{displayName} contents{html}
        }}}""" % (count, offset)
        rows = client.get(source["url"] + "/graphql", {"query": query})["data"]["posts"]["results"]
        if not rows:
            break
        for row in rows:
            if (row.get("baseScore") or 0) < source.get("min_score", 0):
                continue
            yield {
                "id": row["_id"], "title": row["title"], "url": row["pageUrl"],
                "author": (row.get("user") or {}).get("displayName", "Unknown"),
                "date": row.get("postedAt"), "score": row.get("baseScore"),
                "alignment_forum": bool(row.get("af")),
                "text": clean_html((row.get("contents") or {}).get("html")),
            }
        print(f"{source['name']}: fetched {offset + len(rows)}", flush=True)
        if len(rows) < count:
            break


def fetch_wordpress(client, source, limit):
    for offset in range(0, limit, 100):
        data = client.get(
            f"https://public-api.wordpress.com/rest/v1.1/sites/{source['site']}/posts/",
            {"number": min(100, limit - offset), "offset": offset, "order": "DESC"},
        )
        rows = data.get("posts", [])
        for row in rows:
            if row.get("status") != "publish" or row.get("type") != "post":
                continue
            yield {
                "id": str(row["ID"]), "title": html.unescape(row["title"]),
                "url": row["URL"], "author": row["author"]["name"], "date": row["date"],
                "score": None, "text": clean_html(row.get("content")),
            }
        print(f"{source['name']}: fetched {offset + len(rows)}", flush=True)
        if len(rows) < min(100, limit - offset):
            break


def fetch_wordpress_v2(client, source, limit):
    for offset in range(0, limit, 50):
        rows = client.get(source["url"] + "/wp-json/wp/v2/posts", {
            "per_page": min(50, limit - offset), "offset": offset,
            "_fields": "id,title,link,date,content,status,type",
        })
        for row in rows:
            if row.get("content", {}).get("protected"):
                continue
            yield {
                "id": str(row["id"]), "title": clean_html(row["title"]["rendered"]),
                "url": row["link"], "author": source["author"], "date": row["date"],
                "score": None, "text": clean_html(row["content"]["rendered"]),
            }
        print(f"{source['name']}: fetched {offset + len(rows)}", flush=True)
        if len(rows) < min(50, limit - offset):
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--limit", type=int, help="Override each source limit, e.g. 5 for a live smoke check")
    parser.add_argument("--source", help="Fetch only this configured source")
    args = parser.parse_args()
    cfg = config(args.config)
    client = Client("data/cache")
    selected = [s for s in cfg["sources"] if not args.source or s["name"] == args.source]
    if not selected:
        parser.error("No matching source")
    for source in selected:
        fetcher = {"forum": fetch_forum, "wordpress": fetch_wordpress,
                   "wordpress_v2": fetch_wordpress_v2}[source["kind"]]
        rows = list(fetcher(client, source, args.limit or source["limit"]))
        if not rows:
            raise RuntimeError(f"No posts returned for {source['name']}")
        now = datetime.now(timezone.utc).isoformat()
        for row in rows:
            row.update(source=source["name"], fetched_at=now, rights="See SOURCES.md; original author retains rights")
        write_jsonl(f"data/raw/{source['name']}.jsonl", rows)
        print(f"Saved {len(rows)} public posts for {source['name']}", flush=True)


if __name__ == "__main__":
    main()
