"""
RBI Grade B Current Affairs Agent - Cloud version (for GitHub Actions)
Runs on GitHub's free servers on a schedule, publishes dashboard.html via GitHub Pages.

Difference from the local Windows version:
- API key comes from an environment variable (GitHub secret), not hardcoded
- Writes dashboard.html into docs/ folder (what GitHub Pages serves)
- No popup.hta (Windows-only) - this is the "check from phone" companion,
  your laptop's popup automation stays separate for desktop use
- Database (rbi_agent.db) is committed back to the repo so it persists between runs
"""

import feedparser
import sqlite3
import json
import os
import re
from datetime import datetime
from groq import Groq

# ====== CONFIG ======
GROQ_API_KEY = os.environ["GROQ_API_KEY"]   # comes from GitHub secret, set via workflow
GROQ_MODEL = "llama-3.3-70b-versatile"
DB_PATH = "rbi_agent.db"
BATCH_SIZE = 8
MAX_TOKENS = 4000
OUTPUT_DIR = "docs"

RSS_FEEDS = {
    "RBI Press Releases": "https://www.rbi.org.in/pressreleases_rss.xml",
    "RBI Notifications": "https://www.rbi.org.in/notifications_rss.xml",
    "PIB": "https://www.pib.gov.in/ViewRss.aspx?reg=1&lang=1",
    "Economic Times Economy": "https://economictimes.indiatimes.com/news/economy/rssfeeds/1373380680.cms",
    "LiveMint Economy": "https://www.livemint.com/rss/economy",
    "Business Standard Economy": "https://www.business-standard.com/rss/economy-policy-102.rss",
}

CATEGORIES = [
    "Economy", "Banking", "Monetary Policy", "Fiscal Policy",
    "Government Schemes", "Reports and Indices", "International Organizations",
    "Financial Markets", "Insurance", "Agriculture & Rural Development",
    "Sustainable Development", "Miscellaneous Exam-Relevant Topics"
]

client = Groq(api_key=GROQ_API_KEY)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            title TEXT UNIQUE,
            link TEXT,
            published TEXT,
            fetched_at TEXT,
            relevance_score INTEGER,
            importance TEXT,
            category TEXT,
            summary TEXT,
            why_it_matters TEXT,
            exam_relevance TEXT,
            possible_questions TEXT,
            static_concepts TEXT,
            keywords TEXT,
            revision_note TEXT,
            processed INTEGER DEFAULT 0,
            seen INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    return conn


def fetch_rss():
    items = []
    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:10]:
                items.append({
                    "source": source,
                    "title": entry.get("title", "").strip(),
                    "link": entry.get("link", ""),
                    "published": entry.get("published", str(datetime.now())),
                })
        except Exception as e:
            print(f"  [warn] Failed to fetch {source}: {e}")
    return items


def save_raw_item(conn, item):
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO news (source, title, link, published, fetched_at) VALUES (?, ?, ?, ?, ?)",
            (item["source"], item["title"], item["link"], item["published"], str(datetime.now()))
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def build_batch_prompt(items):
    cat_list = ", ".join(CATEGORIES)
    numbered = "\n".join(f'{i+1}. [id={it[0]}] "{it[1]}" (source: {it[2]})' for i, it in enumerate(items))
    return f"""You are an expert RBI Grade B exam mentor. Below is a numbered list of news headlines.
For EACH headline, analyze its relevance to the RBI Grade B economy/banking/finance syllabus.

Headlines:
{numbered}

Respond ONLY with a valid JSON array, no markdown, no preamble, no explanation outside the JSON.
The array must have exactly {len(items)} objects, one per headline, in the same order, each with these keys:
{{
  "id": the numeric id given in brackets,
  "relevant": true or false (false if not relevant to RBI Grade B syllabus, e.g. entertainment, crime, sports, pure politics),
  "relevance_score": integer 1-10,
  "importance": "Critical" or "High" or "Medium" or "Low",
  "category": one of [{cat_list}],
  "summary": "2-3 sentence summary",
  "why_it_matters": "1-2 sentences",
  "exam_relevance": "1-2 sentences on how this could appear in the exam",
  "possible_questions": "1-2 sample MCQ-style questions",
  "static_concepts": "related static GK concepts, comma separated",
  "keywords": "comma separated key terms",
  "revision_note": "one crisp line for quick revision"
}}"""


def process_one_batch(conn, batch, depth=0):
    c = conn.cursor()
    try:
        prompt = build_batch_prompt(batch)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=MAX_TOKENS,
        )
        text = response.choices[0].message.content.strip()
        text = re.sub(r"^```json|```$", "", text, flags=re.MULTILINE).strip()
        results = json.loads(text)

        for data in results:
            row_id = data.get("id")
            if not data.get("relevant", True):
                c.execute("DELETE FROM news WHERE id = ?", (row_id,))
            else:
                c.execute("""
                    UPDATE news SET
                        relevance_score=?, importance=?, category=?, summary=?,
                        why_it_matters=?, exam_relevance=?, possible_questions=?,
                        static_concepts=?, keywords=?, revision_note=?, processed=1
                    WHERE id=?
                """, (
                    data.get("relevance_score", 5), data.get("importance", "Medium"),
                    data.get("category", "Miscellaneous Exam-Relevant Topics"),
                    data.get("summary", ""), data.get("why_it_matters", ""),
                    data.get("exam_relevance", ""), data.get("possible_questions", ""),
                    data.get("static_concepts", ""), data.get("keywords", ""),
                    data.get("revision_note", ""), row_id
                ))
        conn.commit()
        print(f"  Batch of {len(batch)} succeeded (depth {depth}).")
    except Exception as e:
        if len(batch) == 1:
            print(f"  [warn] Single item id={batch[0][0]} failed permanently: {e}")
            c.execute("UPDATE news SET processed = 1, summary = 'PROCESSING_FAILED' WHERE id = ?", (batch[0][0],))
            conn.commit()
        else:
            print(f"  [warn] Batch of {len(batch)} failed ({e}); splitting and retrying...")
            mid = len(batch) // 2
            process_one_batch(conn, batch[:mid], depth + 1)
            process_one_batch(conn, batch[mid:], depth + 1)


def process_batch_with_groq(conn):
    c = conn.cursor()
    c.execute("SELECT id, title, source FROM news WHERE processed = 0")
    rows = c.fetchall()
    if not rows:
        print("No new items to process.")
        return
    print(f"Processing {len(rows)} new items with Groq, in batches of {BATCH_SIZE}...")
    for batch in chunk(rows, BATCH_SIZE):
        process_one_batch(conn, batch)


def generate_dashboard(conn):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    c = conn.cursor()
    c.execute("""
        SELECT title, source, category, importance, relevance_score, summary,
               why_it_matters, exam_relevance, revision_note, link, fetched_at
        FROM news WHERE processed=1 AND summary != 'PROCESSING_FAILED'
        ORDER BY fetched_at DESC LIMIT 100
    """)
    rows = c.fetchall()

    importance_color = {"Critical": "#dc2626", "High": "#ea580c", "Medium": "#ca8a04", "Low": "#65a30d"}

    cards = ""
    for r in rows:
        title, source, category, importance, score, summary, why, exam, note, link, fetched = r
        color = importance_color.get(importance, "#6b7280")
        cards += f"""
        <div class="card">
            <div class="card-header">
                <span class="badge" style="background:{color}">{importance}</span>
                <span class="score">Score: {score}/10</span>
                <span class="category">{category}</span>
            </div>
            <h3><a href="{link}" target="_blank">{title}</a></h3>
            <p class="source">{source} · {fetched[:16]}</p>
            <p><strong>Summary:</strong> {summary}</p>
            <p><strong>Why it matters:</strong> {why}</p>
            <p><strong>Exam relevance:</strong> {exam}</p>
            <p class="note">📌 {note}</p>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RBI Grade B Agent</title>
<style>
body {{ background:#0f172a; color:#e2e8f0; font-family:Segoe UI,Arial,sans-serif; margin:0; padding:16px; max-width:800px; margin:0 auto; }}
h1 {{ color:#38bdf8; font-size:22px; }}
.card {{ background:#1e293b; border-radius:10px; padding:16px; margin-bottom:16px; border-left:4px solid #38bdf8; }}
.card h3 {{ margin:8px 0; font-size:16px; }}
.card h3 a {{ color:#f1f5f9; text-decoration:none; }}
.card h3 a:hover {{ text-decoration:underline; }}
.card-header {{ display:flex; gap:8px; align-items:center; font-size:11px; flex-wrap:wrap; }}
.badge {{ padding:2px 8px; border-radius:4px; color:white; font-weight:bold; }}
.score {{ color:#94a3b8; }}
.category {{ background:#334155; padding:2px 8px; border-radius:4px; }}
.source {{ color:#64748b; font-size:12px; }}
.note {{ color:#fbbf24; font-style:italic; }}
</style></head>
<body>
<h1>RBI Grade B Current Affairs Dashboard</h1>
<p>Last updated: {datetime.now().strftime('%d %b %Y, %I:%M %p')} · {len(rows)} items</p>
{cards}
</body></html>"""

    with open(os.path.join(OUTPUT_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Dashboard updated: {OUTPUT_DIR}/index.html ({len(rows)} items)")


def main():
    print(f"=== RBI Agent cloud run: {datetime.now()} ===")
    conn = init_db()

    print("Fetching RSS feeds...")
    items = fetch_rss()
    new_count = 0
    for item in items:
        if item["title"] and save_raw_item(conn, item):
            new_count += 1
    print(f"  {new_count} new items found out of {len(items)} fetched.")

    process_batch_with_groq(conn)
    generate_dashboard(conn)
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
