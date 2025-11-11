#!/usr/bin/env python3
"""
vinted_bot.py
Un fichier unique qui combine :
 - un collector minimal (requests + BeautifulSoup) pour scanner une page Vinted,
 - un moteur de scoring (profit, velocity, risk, score final),
 - un petit serveur FastAPI exposant /score et /scan,
 - des tests pytest à la fin.

Usage rapide :
  pip install -r requirements.txt
  python vinted_bot.py         # lance l'API sur 0.0.0.0:8000
  # exemple curl (scorer une annonce)
  curl -X POST "http://localhost:8000/score" -H "Content-Type: application/json" \
    -d '{"asking_price":25,"historical_prices":[40,35,38],"likes":3,"views":20}'

IMPORTANT:
 - Vérifie les Terms of Service de Vinted avant d’automatiser du scraping ou d’achats.
 - Ce collector est un *prototype* : adapte les sélecteurs CSS / la logique de parsing en fonction du HTML réel.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import math
import time
import threading
import json
import re

# HTTP / parsing
import requests
from bs4 import BeautifulSoup

# API
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import uvicorn

# Tests
import pytest

# ----------------------
# Config / Hyperparams
# ----------------------
USER_AGENT = "Mozilla/5.0 (compatible; VintedBot/1.0; +https://example.com/bot)"
REQUESTS_TIMEOUT = 10  # seconds
MIN_SECONDS_BETWEEN_REQUESTS = 1.2  # rate limit between HTTP calls (polite)

VINTED_FEES_PCT = 0.10
AVG_SHIPPING_COST = 5.0
DEFAULT_REFURB_COST = 3.0

MIN_PROFIT_DESIRABLE = 5.0
MAX_PROFIT_CONSIDERED = 200.0

WEIGHT_PROFIT = 0.40
WEIGHT_VELOCITY = 0.55
WEIGHT_RISK = 0.15

# Collector settings
DEFAULT_HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8"}
SESSION = requests.Session()
SESSION.headers.update(DEFAULT_HEADERS)

# -------------
# Utilities
# -------------
def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

def median(nums: List[float]) -> Optional[float]:
    if not nums:
        return None
    s = sorted(nums)
    n = len(s)
    if n % 2 == 1:
        return float(s[n // 2])
    return float((s[n//2 - 1] + s[n//2]) / 2.0)

# -------------
# Scoring Engine
# -------------
def estimate_market_price(historical_prices: List[float]) -> Optional[float]:
    return median(historical_prices)

def compute_net_profit(asking_price: float,
                       market_price_est: Optional[float],
                       fees_pct: float = VINTED_FEES_PCT,
                       shipping: float = AVG_SHIPPING_COST,
                       refurb: float = DEFAULT_REFURB_COST) -> Optional[float]:
    if market_price_est is None:
        return None
    gross = market_price_est - asking_price
    fees = market_price_est * fees_pct
    net = gross - fees - shipping - refurb
    return round(net, 2)

def velocity_score(asking_price: float,
                   market_price_est: Optional[float],
                   posted_at: Optional[str] = None,
                   likes: int = 0,
                   views: int = 0,
                   brand_popularity: float = 0.5) -> float:
    if market_price_est is None:
        return 0.0
    price_ratio = asking_price / market_price_est if market_price_est > 0 else 1.0
    price_factor = clamp(1.5 - price_ratio, 0.0, 1.5) / 1.5

    recency_factor = 0.0
    if posted_at:
        try:
            dt = datetime.fromisoformat(posted_at.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            delta_hours = max(0.0, (now - dt).total_seconds() / 3600.0)
            recency_factor = clamp(1 - (delta_hours / (24 * 7)), 0.0, 1.0)
        except Exception:
            recency_factor = 0.0

    social = math.log(1 + likes + views / 20.0) / 5.0
    social = clamp(social, 0.0, 1.0)
    brand = clamp(brand_popularity, 0.0, 1.0)

    score = 0.5 * price_factor + 0.3 * recency_factor + 0.15 * social + 0.05 * brand
    return clamp(score, 0.0, 1.0)

def risk_penalty(photo_quality: float = 1.0,
                 ambiguous_brand: bool = False,
                 suspect_low_price: bool = False) -> float:
    r = 0.0
    if photo_quality < 0.5:
        r += 0.25
    if ambiguous_brand:
        r += 0.35
    if suspect_low_price:
        r += 0.25
    return clamp(r, 0.0, 1.0)

def final_score(net_profit: Optional[float],
                velocity: float,
                risk: float,
                min_profit: float = MIN_PROFIT_DESIRABLE,
                max_profit: float = MAX_PROFIT_CONSIDERED) -> Dict[str, Any]:
    if net_profit is None:
        profit_norm = 0.0
    else:
        profit_norm = (net_profit - min_profit) / (max_profit - min_profit)
        profit_norm = clamp(profit_norm, 0.0, 1.0)

    raw = WEIGHT_PROFIT * profit_norm + WEIGHT_VELOCITY * velocity - WEIGHT_RISK * risk
    raw = clamp(raw, 0.0, 1.0)
    score_pct = round(raw * 100, 2)

    return {
        "score": score_pct,
        "profit_normalized": round(profit_norm, 4),
        "velocity": round(velocity, 4),
        "risk": round(risk, 4),
        "net_profit": None if net_profit is None else round(net_profit, 2)
    }

def score_listing(listing: Dict[str, Any]) -> Dict[str, Any]:
    asking = float(listing.get("asking_price", 0.0))
    market_est = listing.get("market_price_est")
    if market_est is None:
        market_est = estimate_market_price(listing.get("historical_prices", []))
    netp = compute_net_profit(asking, market_est)
    vel = velocity_score(asking, market_est,
                         posted_at=listing.get("posted_at"),
                         likes=int(listing.get("likes", 0)),
                         views=int(listing.get("views", 0)),
                         brand_popularity=float(listing.get("brand_popularity", 0.5)))
    risk = risk_penalty(photo_quality=float(listing.get("photo_quality", 1.0)),
                        ambiguous_brand=bool(listing.get("ambiguous_brand", False)),
                        suspect_low_price=bool(listing.get("suspect_low_price", False)))

    final = final_score(netp, vel, risk)
    return {
        "asking_price": asking,
        "market_price_est": market_est,
        "net_profit": final["net_profit"],
        "score": final["score"],
        "components": {
            "profit_normalized": final["profit_normalized"],
            "velocity": final["velocity"],
            "risk": final["risk"]
        },
        "raw": {
            "velocity_raw": vel,
            "risk_raw": risk,
            "net_profit_raw": netp
        }
    }

# ---------------------
# Minimal Collector
# ---------------------
class Collector:
    """
    Collector minimal : fetch une page de recherche Vinted passée en paramètre,
    parse (via BeautifulSoup) un ensemble d'annonces et retourne une liste d'items.

    ATTENTION : Vinted change souvent sa structure HTML. Les sélecteurs ci-dessous
    sont des EXEMPLES. Adapte-les au HTML réel que tu observes.
    """

    def __init__(self, session: requests.Session = SESSION, rate_limit_seconds: float = MIN_SECONDS_BETWEEN_REQUESTS):
        self.session = session
        self.rate_limit = rate_limit_seconds
        self._last_request_ts = 0.0

    def _throttle(self):
        elapsed = time.time() - self._last_request_ts
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)

    def fetch_search_page(self, url: str) -> Optional[str]:
        self._throttle()
        try:
            r = self.session.get(url, timeout=REQUESTS_TIMEOUT)
            self._last_request_ts = time.time()
            if r.status_code == 200:
                return r.text
            else:
                print(f"[collector] HTTP {r.status_code} for {url}")
                return None
        except Exception as e:
            print(f"[collector] request error: {e}")
            return None

    def parse_listings_from_search_html(self, html: str, max_items: int = 30) -> List[Dict[str, Any]]:
        """
        Parse HTML de la page de résultats et extrait une liste d'annonces.
        Retourne des items minimally structured :
         - title, asking_price (float), url, posted_at (ISO str optional), likes, views, thumbnails[]
         - historical_prices = [] (vide) -> optionnel d'enrichir ensuite
        """
        soup = BeautifulSoup(html, "html.parser")
        results = []

        # EXEMPLE de sélecteur : adapte aux classes/attributes réels
        # On cherche des éléments <a> représentant chaque annonce.
        # Sur Vinted il y a souvent des balises <div> avec data-test-id, etc.
        # Ici on tente un selector générique.
        anchors = soup.select("a")  # to be narrowed by real selector
        # heuristique : garder les ancres qui ressemblent à des annonces (url contenant '/item/')
        anchors = [a for a in anchors if a.get("href") and "/item/" in a.get("href")]

        seen = set()
        for a in anchors:
            if len(results) >= max_items:
                break
            href = a.get("href")
            href_full = href if href.startswith("http") else ("https://www.vinted.fr" + href)
            if href_full in seen:
                continue
            seen.add(href_full)

            # Tenter d'extraire titre / prix / image
            title = a.get_text(strip=True) or ""
            price_text = ""
            # chercher dans le noeud des span contenant '€'
            price_span = a.find(lambda tag: tag.name in ["span", "div"] and "€" in tag.get_text() if tag.get_text() else False)
            if price_span:
                price_text = price_span.get_text()
            else:
                # fallback: regex in the anchor text
                m = re.search(r"(\d+[,.]?\d*)\s*€", a.get_text())
                price_text = m.group(0) if m else ""

            asking = 0.0
            if price_text:
                m = re.search(r"(\d+[,.]?\d*)", price_text)
                if m:
                    asking = float(m.group(1).replace(",", "."))

            # images
            thumbs = []
            img = a.find("img")
            if img:
                src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
                if src:
                    thumbs.append(src)

            # posted_at : often not on the anchor, skip (None)
            posted_at = None

            item = {
                "title": title,
                "asking_price": asking,
                "url": href_full,
                "posted_at": posted_at,
                "likes": 0,
                "views": 0,
                "thumbnails": thumbs,
                "historical_prices": []  # enrichment later
            }
            results.append(item)

        return results

    def fetch_and_parse(self, url: str, max_items: int = 30) -> List[Dict[str, Any]]:
        html = self.fetch_search_page(url)
        if not html:
            return []
        return self.parse_listings_from_search_html(html, max_items=max_items)

# ---------------------
# Enricher (placeholder)
# ---------------------
def enrich_listing_with_price_history(listing: Dict[str, Any]) -> Dict[str, Any]:
    """
    Placeholder d'enrichissement :
     - récupérer historique des prix pour l'item (ou modèle) via une source interne/DB
     - détecter brand_popularity via lookup
     - estimer photo_quality automatiquement via heuristique (taille de l'image / présence de watermark etc.)
    Ici on fait des heuristiques simples : si asking_price > 0, propose market_est = asking * 1.5 (exemple),
    mais en production il faut un vrai historique.
    """
    asking = listing.get("asking_price", 0.0)
    if asking and asking > 0:
        # simulate historical median price as 1.5 * asking (very naive)
        market_est = round(asking * 1.5, 2)
        listing["historical_prices"] = [market_est, market_est * 0.9, market_est * 1.1]
        listing["market_price_est"] = estimate_market_price(listing["historical_prices"])
    else:
        listing["historical_prices"] = []
        listing["market_price_est"] = None

    # heuristics for brand popularity & photo quality
    listing["brand_popularity"] = 0.5
    listing["photo_quality"] = 0.9 if listing.get("thumbnails") else 0.4
    listing["ambiguous_brand"] = False
    listing["suspect_low_price"] = (listing["market_price_est"] is not None) and (listing["asking_price"] < max(1.0, 0.2 * listing["market_price_est"]))
    return listing

# ---------------------
# FastAPI endpoints
# ---------------------
app = FastAPI(title="Vinted Buy/Resell Scorer (Prototype)")

class ListingIn(BaseModel):
    asking_price: float
    market_price_est: Optional[float] = None
    historical_prices: Optional[List[float]] = []
    posted_at: Optional[str] = None
    likes: Optional[int] = 0
    views: Optional[int] = 0
    brand_popularity: Optional[float] = 0.5
    photo_quality: Optional[float] = 1.0
    ambiguous_brand: Optional[bool] = False
    suspect_low_price: Optional[bool] = False
    title: Optional[str] = None
    url: Optional[str] = None

@app.post("/score")
def score_endpoint(listing: ListingIn):
    try:
        res = score_listing(listing.dict())
        return {"ok": True, "result": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ScanIn(BaseModel):
    url: str
    max_items: Optional[int] = 30
    top_n: Optional[int] = 10
    enrich: Optional[bool] = True

@app.post("/scan")
def scan_endpoint(body: ScanIn):
    """
    Lance un scan ponctuel d'une page de recherche Vinted (URL).
    Retourne une liste d'annonces scorées, triées par score décroissant.
    NOTE: ce endpoint effectue des requêtes HTTP synchrones et doit être utilisé
    prudemment (rate-limit côté client).
    """
    collector = Collector()
    raw_items = collector.fetch_and_parse(body.url, max_items=body.max_items)
    scored = []
    for it in raw_items:
        if body.enrich:
            it = enrich_listing_with_price_history(it)
        scored_item = score_listing(it)
        # merge metadata
        scored_item["title"] = it.get("title")
        scored_item["url"] = it.get("url")
        scored_item["thumbnails"] = it.get("thumbnails", [])
        scored.append(scored_item)
    # trier par score décroissant
    scored_sorted = sorted(scored, key=lambda x: x["score"], reverse=True)
    return {"ok": True, "count": len(scored_sorted), "results": scored_sorted[:body.top_n]}

# ---------------
# CLI / Runner
# ---------------
def run_api(host: str = "0.0.0.0", port: int = 8000):
    print(f"[api] starting on http://{host}:{port}")
    uvicorn.run("vinted_bot:app", host=host, port=port, reload=False)

# ---------------
# Tests (pytest)
# ---------------
def _make_demo_listing():
    return {
        "asking_price": 25.0,
        "historical_prices": [40, 35, 38, 45, 30],
        "posted_at": datetime.now(timezone.utc).isoformat(),
        "likes": 3, "views": 50,
        "brand_popularity": 0.8,
        "photo_quality": 0.9,
        "ambiguous_brand": False
    }

def test_estimate_market_price():
    assert estimate_market_price([10, 20, 30]) == 20
    assert estimate_market_price([]) is None

def test_compute_net_profit_none():
    assert compute_net_profit(10, None) is None

def test_compute_net_profit_basic():
    net = compute_net_profit(25.0, 40.0, fees_pct=0.1, shipping=5.0, refurb=3.0)
    # gross 15 - fees 4 - shipping 5 - refurb 3 -> 3
    assert net == 3.0

def test_velocity_and_risk_ranges():
    v = velocity_score(25, 40, likes=0, views=0, brand_popularity=0.5)
    assert 0.0 <= v <= 1.0
    r = risk_penalty(photo_quality=0.2, ambiguous_brand=True, suspect_low_price=False)
    assert 0.0 <= r <= 1.0

def test_final_score_output():
    fs = final_score(10.0, velocity=0.8, risk=0.1)
    assert "score" in fs and 0 <= fs["score"] <= 100

def test_score_listing_full():
    out = score_listing(_make_demo_listing())
    assert "score" in out and "net_profit" in out

# -------------
# Entrypoint
# -------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Vinted bot (API + collector + scoring)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--run-tests", action="store_true", help="Run pytest unit tests and exit")
    args = parser.parse_args()
    if args.run_tests:
        # lancer pytest programmatically
        raise SystemExit(pytest.main([__file__, "-q"]))
    else:
        run_api(host=args.host, port=args.port)
