"""
Job Scanner — scrapes LinkedIn, Indeed, Google Jobs for Germany
and scores results using Claude API.

Requirements:
    pip install python-jobspy anthropic python-dotenv

Environment variables (put in .env or GitHub Actions secrets):
    ANTHROPIC_API_KEY=...

Usage:
    python job_scanner.py
"""

import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
import anthropic
import pandas as pd

# Try importing JobSpy
try:
    from jobspy import scrape_jobs
except ImportError:
    raise ImportError("Run: pip install python-jobspy")

load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Hard filter — exclude any job whose title contains these (case-insensitive)
EXCLUDE_TITLE_TERMS = [
    "intern", "internship", "working student", "werkstudent",
    "junior", "assistant", "coordinator", "sdr", "bdr",
    "engineer", "developer", "data scientist", "software",
    "devops", "backend", "frontend", "fullstack", "full-stack",
]

# All search terms to run — JobSpy runs each separately, results are merged + deduped
SEARCH_TERMS = [
    "marketing manager",
    "brand manager",
    "go-to-market",
    "growth manager",
    "product marketing manager",
    "customer marketing",
    "partnerships manager",
    "channel partnerships",
    "AI strategy",
    "AI enablement",
    "AI solutions manager",
    "market expansion",
    "expansion manager",
    "venture development",
    "commercial strategy",
    "strategy manager",
    "market development",
    "customer success manager",
    "solutions manager",
    "innovation manager",
]

# ─────────────────────────────────────────────
# SCORING PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise job relevance scorer for a senior marketing and GTM professional 
seeking mid-to-senior roles in Germany. You follow scoring criteria exactly and return only valid JSON."""

def build_scoring_prompt(job: dict) -> str:
    title = job.get("title", "")
    company = job.get("company", "")
    location = job.get("location", "")
    description = (job.get("description") or "")[:4000]
    applicants = job.get("num_urgent_words") or job.get("job_function") or "unknown"
    date_posted = job.get("date_posted", "unknown")

    return f"""Score this job for a senior marketing / GTM professional seeking mid-to-senior roles in Germany.

JOB DETAILS:
Title: {title}
Company: {company}
Location: {location}
Date Posted: {date_posted}
Applicant Count: {applicants}
Description:
{description}

---
SCORING RULES (add up all applicable points):

LOCATION (pick highest applicable):
- Frankfurt / Rhine-Main area → +5
- Munich, Cologne, Düsseldorf, Hamburg, Berlin, Wiesbaden → +2
- Rest of Germany, or fully remote from Germany → +1
- EU-remote-only (no German base) → 0 (and flag as excluded)

LANGUAGE:
- Job description written primarily in English → +3
- German only → +0

SENIORITY (title contains any of these words):
- Head, Director, Senior, Lead, Principal, Global, VP → +3

APPLICANT COUNT:
- Fewer than 50 applicants → +3
- Fewer than 200 applicants or applicant count unknown → +2
- 200 or more applicants → +1

INTERNATIONAL SCOPE (description mentions any):
- Global, International, EMEA, Worldwide → +2

ROLE FIT — Core clusters (award +3 if clearly matches any):
- Marketing leadership: Marketing, Brand, Growth, Go-to-Market, Product Marketing, Customer Marketing
- Business-facing AI: AI Strategy, AI Adoption, AI Enablement, AI Solutions (non-technical)
- Strategic expansion / partnerships: Expansion, Venture Development, Partnerships, Channel Partnerships, Customer Success (non-technical)

ROLE FIT — Hidden opportunity (award +3 if adjacent title but description strongly overlaps with above):
- Adjacent titles: Strategy Manager, Market Development Manager, Growth Manager, Commercial Strategy Manager, 
  Innovation Manager, Solutions Manager, Partnerships Manager, Expansion Manager, Customer Success Manager
- Only award if NOT primarily sales-only, technical delivery, support-only, or junior

COMPANY BOOST:
- OpenAI, Anthropic, Mistral → +10
- Premium tech (e.g. Google, Apple, Meta, Stripe, Figma, Notion), luxury/premium consumer brands, 
  top-tier consulting (McKinsey, BCG, Bain), premium consumer/retail → +3

DEALBREAKERS (flag and set score to 0):
- Role is EU-remote-only with no German base required
- Role is clearly a pure sales/quota-carrying role
- Role is clearly a technical engineering role
- Role is junior despite title appearing senior
- Requires commission-only or heavily commission-based pay

---
Return ONLY a valid JSON object — no markdown, no explanation outside the JSON:
{{
  "score": <integer 0-30>,
  "role_type": "<one of: Marketing Leadership | Business-Facing AI | Strategic Expansion & Partnerships | Adjacent/Hidden Opportunity | No Match>",
  "reason": "<2 sentences max explaining the score>",
  "dealbreaker": <true or false>,
  "dealbreaker_reason": "<short reason if dealbreaker is true, else null>",
  "highlights": ["<up to 3 short strings — most relevant signals from the description>"]
}}"""


# ─────────────────────────────────────────────
# SCRAPING
# ─────────────────────────────────────────────

def scrape_all_jobs() -> list[dict]:
    """Run all search terms across LinkedIn, Indeed, Google Jobs."""
    all_jobs = []
    seen_ids = set()

    client_sources = ["linkedin", "indeed", "google"]

    for term in SEARCH_TERMS:
        print(f"  Scraping: '{term}'...")
        try:
            df = scrape_jobs(
                site_name=client_sources,
                search_term=term,
                location="Germany",
                results_wanted=30,      # per source per term
                hours_old=72,           # last 3 days only
                country_indeed="Germany",
                linkedin_fetch_description=True,
            )
            if df is None or df.empty:
                continue

            records = df.to_dict("records")
            for job in records:
                job_id = job.get("id") or f"{job.get('company','')}_{job.get('title','')}_{job.get('location','')}"
                if job_id not in seen_ids:
                    seen_ids.add(job_id)
                    all_jobs.append(job)

        except Exception as e:
            print(f"    Warning: failed for '{term}': {e}")

        time.sleep(2)  # polite delay between searches

    print(f"\n  Total unique raw jobs scraped: {len(all_jobs)}")
    return all_jobs


# ─────────────────────────────────────────────
# HARD FILTERING
# ─────────────────────────────────────────────

def hard_filter(jobs: list[dict]) -> list[dict]:
    """Remove obvious mismatches before sending to Claude."""
    filtered = []
    for job in jobs:
        title = (job.get("title") or "").lower()

        # Exclude bad title terms
        if any(term in title for term in EXCLUDE_TITLE_TERMS):
            continue

        # Must have a description (otherwise Claude can't score meaningfully)
        if not job.get("description"):
            continue

        filtered.append(job)

    print(f"  After hard filter: {len(filtered)} jobs remain")
    return filtered


# ─────────────────────────────────────────────
# CLAUDE SCORING
# ─────────────────────────────────────────────

def score_jobs(jobs: list[dict]) -> list[dict]:
    """Score each job using Claude Haiku via the Anthropic API."""
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set in environment")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    scored = []

    for i, job in enumerate(jobs):
        print(f"  Scoring {i+1}/{len(jobs)}: {job.get('title')} @ {job.get('company')}")
        try:
            message = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=512,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": build_scoring_prompt(job)}
                ],
            )

            raw = message.content[0].text.strip()

            # Strip markdown fences if model adds them anyway
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            result = json.loads(raw)

            job["ai_score"] = result.get("score", 0)
            job["ai_role_type"] = result.get("role_type", "")
            job["ai_reason"] = result.get("reason", "")
            job["ai_dealbreaker"] = result.get("dealbreaker", False)
            job["ai_dealbreaker_reason"] = result.get("dealbreaker_reason")
            job["ai_highlights"] = result.get("highlights", [])

        except json.JSONDecodeError as e:
            print(f"    JSON parse error for job {i+1}: {e}")
            job["ai_score"] = 0
            job["ai_dealbreaker"] = False
            job["ai_reason"] = "Scoring failed"
            job["ai_highlights"] = []

        except Exception as e:
            print(f"    Scoring error for job {i+1}: {e}")
            job["ai_score"] = 0
            job["ai_dealbreaker"] = False
            job["ai_reason"] = "Scoring failed"
            job["ai_highlights"] = []

        time.sleep(0.3)  # stay within rate limits

    return scored_and_sorted(scored if scored else jobs)


def scored_and_sorted(jobs: list[dict]) -> list[dict]:
    """Remove dealbreakers and sort by score descending."""
    valid = [j for j in jobs if not j.get("ai_dealbreaker", False)]
    return sorted(valid, key=lambda x: x.get("ai_score", 0), reverse=True)


# ─────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────

def print_summary(jobs: list[dict], top_n: int = 20):
    """Print a readable ranked summary to stdout."""
    print(f"\n{'='*70}")
    print(f"TOP {min(top_n, len(jobs))} JOBS — {datetime.now().strftime('%Y-%m-%d')}")
    print(f"{'='*70}\n")

    for i, job in enumerate(jobs[:top_n], 1):
        highlights = " | ".join(job.get("ai_highlights") or [])
        print(f"#{i:02d}  [{job.get('ai_score', 0):>2} pts]  {job.get('title')} @ {job.get('company')}")
        print(f"      {job.get('location')}  •  {job.get('ai_role_type')}")
        print(f"      {job.get('ai_reason')}")
        if highlights:
            print(f"      ✦ {highlights}")
        print(f"      🔗 {job.get('job_url', 'No URL')}")
        print()


def save_to_json(jobs: list[dict], path: str = "scored_jobs.json"):
    """Save full results to JSON for downstream use (email, Sheets, etc.)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(jobs, f, ensure_ascii=False, indent=2, default=str)
    print(f"  Saved {len(jobs)} scored jobs to {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n── Step 1: Scraping jobs ──")
    raw_jobs = scrape_all_jobs()

    print("\n── Step 2: Hard filtering ──")
    filtered_jobs = hard_filter(raw_jobs)

    if not filtered_jobs:
        print("No jobs passed the hard filter today.")
        return

    print(f"\n── Step 3: Scoring {len(filtered_jobs)} jobs with Claude ──")
    scored_jobs = score_jobs(filtered_jobs)

    print("\n── Step 4: Results ──")
    print_summary(scored_jobs, top_n=20)
    save_to_json(scored_jobs)

    print("\n── Done. Next steps: add deduplication (Google Sheets) + email digest ──")


if __name__ == "__main__":
    main()
