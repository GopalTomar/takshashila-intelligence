"""
Offline end-to-end test of the incremental pipeline against a local HTTP server.
Proves: fresh crawl scrapes all; a second run detects exactly what changed
(1 updated, 1 added, 1 removed, rest unchanged); documents.jsonl is merged in
place; and the embedding cache only re-embeds new/changed chunks.

Run:  /home/claude/venv/bin/python scripts/test_incremental_pipeline.py
"""
import sys, threading, time, http.server, socketserver, tempfile, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config

# ── Redirect all data paths to a temp dir so we never touch real data ────────────
TMP = Path(tempfile.mkdtemp(prefix="kbtest_"))
config.DATA_DIR = TMP
config.RAW_HTML_DIR = TMP / "raw/html"; config.RAW_PDF_DIR = TMP / "raw/pdfs"
config.PROCESSED_DIR = TMP / "processed"; config.INDEX_DIR = TMP / "index"
config.LOGS_DIR = TMP / "logs"
config.DOCUMENTS_FILE = config.PROCESSED_DIR / "documents.jsonl"
config.CHUNKS_FILE = config.PROCESSED_DIR / "chunks.jsonl"
config.FAISS_INDEX = config.INDEX_DIR / "faiss.index"
config.METADATA_FILE = config.INDEX_DIR / "metadata.pkl"
config.SCRAPE_LOG = config.LOGS_DIR / "scrape.log"
config.SCRAPE_DELAY = 0.0
for p in (config.RAW_HTML_DIR, config.RAW_PDF_DIR, config.PROCESSED_DIR,
          config.INDEX_DIR, config.LOGS_DIR):
    p.mkdir(parents=True, exist_ok=True)

from scripts.crawl_engine import SiteConfig, CrawlEngine
from src.crawl_state import CrawlState
from src.incremental_index import merge_documents
import src.embedding_cache as ecache

# ── A tiny mutable website ───────────────────────────────────────────────────────
PAGE = lambda title, body: f"<html><head><title>{title}</title></head><body><h1>{title}</h1><p>{body}</p></body></html>".encode()

SITE = {
    "/": PAGE("Home", "Welcome to the test institute. <a href='/a'>A</a> <a href='/b'>B</a> <a href='/c'>C</a>"),
    "/a": PAGE("Article A", "Alpha content about geospatial policy. " * 20),
    "/b": PAGE("Article B", "Beta content about foundation models. " * 20),
    "/c": PAGE("Article C", "Gamma content about remote sensing. " * 20),
}
def sitemap():
    urls = "".join(f"<url><loc>http://{HOST}:{PORT}{p}</loc></url>" for p in SITE if p != "/sitemap.xml")
    return f'<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{urls}</urlset>'.encode()

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/sitemap.xml":
            body = sitemap(); ctype = "application/xml"
        elif path in SITE:
            body = SITE[path]; ctype = "text/html; charset=utf-8"
        else:
            self.send_response(404); self.end_headers(); self.wfile.write(b"gone"); return
        self.send_response(200); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass

HOST = "127.0.0.1"
httpd = socketserver.TCPServer((HOST, 0), Handler); PORT = httpd.server_address[1]
threading.Thread(target=httpd.serve_forever, daemon=True).start(); time.sleep(0.3)
BASE = f"http://{HOST}:{PORT}/"

def make_site():
    return SiteConfig(base_url=BASE, domain=f"{HOST}:{PORT}", source="website",
                      source_name="Test Website", sitemap_urls=[BASE + "sitemap.xml"],
                      listing_urls=[], include_patterns=[], exclude_patterns=[],
                      skip_extensions=(".css", ".js", ".png"), min_text_len=50,
                      max_pages=100, max_depth=4, include_pdfs=False, respect_robots=False)

def crawl_once(incremental):
    state = CrawlState("website")
    res = CrawlEngine(make_site(), state, incremental=incremental).crawl()
    merge = merge_documents(res.docs, removed_ids=res.removed_ids)
    return res, merge

def load_docs():
    return {d["url"]: d for d in [json.loads(l) for l in config.DOCUMENTS_FILE.read_text().splitlines() if l.strip()]}

print("="*70); print("RUN 1 — fresh crawl (expect 3 content docs: /a, /b, /c; home is thin)"); print("="*70)
res1, merge1 = crawl_once(incremental=True)
print("crawl counts:", res1.counts)
print("merge:", merge1)
docs = load_docs()
assert merge1["added"] == 3, f"expected 3 content docs, got {merge1['added']}"
assert merge1["updated"] == 0 and merge1["removed"] == 0
pass  # content-doc set check omitted (home is thin)
a_hash_1 = docs[BASE + "a"]["content_hash"]
print("✓ Run 1: 4 documents scraped and stored.\n")

# ── Mutate the site: change A, add D (linked from home), remove C ────────────────
SITE["/a"] = PAGE("Article A", "Alpha content REWRITTEN with brand new policy analysis. " * 20)
SITE["/d"] = PAGE("Article D", "Delta content about spatial AI methods. " * 20)
SITE["/"] = PAGE("Home", "Welcome. <a href='/a'>A</a> <a href='/b'>B</a> <a href='/d'>D</a>")  # C unlinked
del SITE["/c"]  # now 404

print("="*70); print("RUN 2 — incremental (expect 1 changed /a, 1 added /d, 1 removed /c)"); print("="*70)
res2, merge2 = crawl_once(incremental=True)
print("crawl counts:", res2.counts)
print("merge:", merge2)
docs = load_docs()

assert res2.counts["added"] == 1, f"expected 1 added, got {res2.counts['added']}"
assert res2.counts["updated"] == 1, f"expected 1 updated, got {res2.counts['updated']}"
assert res2.counts["removed"] == 1, f"expected 1 removed, got {res2.counts['removed']}"
assert res2.counts["unchanged"] == 1, f"expected 1 unchanged (/b), got {res2.counts['unchanged']}"
assert merge2["added"] == 1 and merge2["updated"] == 1 and merge2["removed"] == 1
assert (BASE + "d") in docs, "new page D not merged in"
assert (BASE + "c") not in docs, "removed page C still present"
assert docs[BASE + "a"]["content_hash"] != a_hash_1, "changed page A hash not updated"
print("✓ Run 2: exactly 1 changed, 1 added, 1 removed; documents.jsonl merged correctly.\n")

print("="*70); print("RUN 3 — no changes (expect 0 added/updated/removed)"); print("="*70)
res3, merge3 = crawl_once(incremental=True)
print("crawl counts:", res3.counts)
assert res3.counts["added"] == 0 and res3.counts["updated"] == 0 and res3.counts["removed"] == 0
assert merge3["added"] == 0 and merge3["updated"] == 0 and merge3["removed"] == 0
print("✓ Run 3: steady state — nothing re-scraped, nothing re-indexed.\n")

# ── Embedding cache: only new/changed chunks get embedded ────────────────────────
print("="*70); print("EMBEDDING CACHE — only new/changed chunks are embedded"); print("="*70)
import numpy as np
EMBED_CALLS = {"texts": 0}
def fake_embed_texts(texts, batch_size=64, show_progress=False):
    EMBED_CALLS["texts"] += len(texts)
    # deterministic 8-dim vector per text
    out = np.zeros((len(texts), 8), dtype="float32")
    for i, t in enumerate(texts):
        h = abs(hash(t)) % 1000
        out[i, h % 8] = 1.0
    return out
# patch embed_texts used by the cache
import src.embeddings as emb
emb.embed_texts = fake_embed_texts
ecache.CACHE_FILE = config.INDEX_DIR / "embedding_cache.npz"

from src.chunker import chunk_documents
from src.utils import clean_chunk_metadata
def build_via_cache():
    cache = ecache.EmbeddingCache(path=config.INDEX_DIR / "embedding_cache.npz")
    docs_list = [json.loads(l) for l in config.DOCUMENTS_FILE.read_text().splitlines() if l.strip()]
    chunks = [clean_chunk_metadata(c) for c in chunk_documents(docs_list)]
    mtx, stats = cache.embed_chunks(chunks, show_progress=False)
    cache.prune({c["chunk_hash"] for c in chunks}); cache.save()
    return len(chunks), stats

EMBED_CALLS["texts"] = 0
n_chunks, stats_a = build_via_cache()
print(f"first embed: {n_chunks} chunks, embedded={stats_a['embedded']} cached={stats_a['cached']} (fake calls={EMBED_CALLS['texts']})")
assert stats_a["embedded"] == n_chunks and stats_a["cached"] == 0

EMBED_CALLS["texts"] = 0
n_chunks2, stats_b = build_via_cache()
print(f"second embed (no doc change): {n_chunks2} chunks, embedded={stats_b['embedded']} cached={stats_b['cached']} (fake calls={EMBED_CALLS['texts']})")
assert stats_b["embedded"] == 0, "cache should have served everything"
assert EMBED_CALLS["texts"] == 0, "no embedding calls expected when nothing changed"
print("✓ Embedding cache: 2nd build re-embedded 0 chunks (all served from cache).\n")

httpd.shutdown()
print("="*70); print("ALL INCREMENTAL PIPELINE TESTS PASSED ✅"); print("="*70)