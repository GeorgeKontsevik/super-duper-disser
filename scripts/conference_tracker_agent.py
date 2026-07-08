#!/usr/bin/env python3
"""Search, verify, and append reviewable conference tracker candidates."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import html
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
REVIEW_HEADERS = [
    "Human verified",
    "agent_run_id",
    "agent_checked_at",
    "agent_source_row",
]

YEAR_FIELDS = [
    "status",
    "candidate_site",
    "confidence",
    "evidence_url",
    "evidence_title",
    "evidence_snippet",
    "notes",
]

COMPACT_2027_HEADERS = [
    "Human verified",
    "Year",
    "Conference",
    "CORE23 Ranking",
    "CORE link",
    "Dates 2027",
    "City",
    "Country",
    "Verified site",
]


@dataclasses.dataclass(frozen=True)
class SourceRow:
    row_number: int
    headers: list[str]
    values: list[str]
    name: str
    current_site: str = ""


@dataclasses.dataclass(frozen=True)
class SearchResult:
    url: str
    title: str
    snippet: str = ""


@dataclasses.dataclass(frozen=True)
class VerifiedCandidate:
    source: SourceRow
    year: int
    site: str
    evidence_url: str
    evidence_title: str
    evidence_snippet: str
    confidence: str
    notes: str


def utc_stamp() -> tuple[str, str]:
    now = dt.datetime.now(dt.UTC).replace(microsecond=0)
    return now.strftime("%Y%m%dT%H%M%SZ"), now.isoformat().replace("+00:00", "Z")


def _cell(row_by_header: dict[str, str], *names: str) -> str:
    lowered = {key.strip().lower(): value.strip() for key, value in row_by_header.items()}
    for name in names:
        value = lowered.get(name.strip().lower())
        if value:
            return value
    return ""


def normalize_source_row(headers: list[str], values: list[str], row_number: int) -> SourceRow:
    padded = values + [""] * max(0, len(headers) - len(values))
    row_by_header = dict(zip(headers, padded, strict=False))
    return SourceRow(
        row_number=row_number,
        headers=headers,
        values=padded[: len(headers)],
        name=_cell(row_by_header, "Column 1", "conference_name", "Conference", "Name"),
        current_site=_cell(row_by_header, "Dates / Location source", "Main Conference Website", "source"),
    )


def parse_source_rows(values: list[list[str]], limit: int | None) -> tuple[list[str], list[SourceRow]]:
    if not values:
        return [], []
    headers = values[0]
    rows: list[SourceRow] = []
    for offset, values_row in enumerate(values[1:], start=2):
        row = normalize_source_row(headers, values_row, row_number=offset)
        if row.name:
            rows.append(row)
        if limit and len(rows) >= limit:
            break
    return headers, rows


def build_review_headers(source_headers: list[str], years: list[int]) -> list[str]:
    year_headers = [f"agent_{field}_{year}" for year in years for field in YEAR_FIELDS]
    return source_headers + REVIEW_HEADERS + year_headers


def build_review_row(
    source_headers: list[str],
    source: SourceRow,
    candidates_by_year: dict[int, VerifiedCandidate | None],
    statuses_by_year: dict[int, str],
    years: list[int],
    run_id: str,
    checked_at: str,
) -> list[str]:
    row = source.values + [""] * max(0, len(source_headers) - len(source.values))
    row = row[: len(source_headers)]
    review_values = [
        "FALSE",
        run_id,
        checked_at,
        str(source.row_number),
    ]
    year_values: list[str] = []
    for year in years:
        candidate = candidates_by_year.get(year)
        if candidate:
            year_values.extend(
                [
                    statuses_by_year.get(year, "verified"),
                    candidate.site,
                    candidate.confidence,
                    candidate.evidence_url,
                    candidate.evidence_title,
                    candidate.evidence_snippet,
                    candidate.notes,
                ]
            )
        else:
            year_values.extend([statuses_by_year.get(year, "no_verified_candidate"), "", "", "", "", "", ""])
    return row + review_values + year_values


def build_compact_2027_headers() -> list[str]:
    return COMPACT_2027_HEADERS[:]


def source_value(source_headers: list[str], source: SourceRow, *names: str) -> str:
    return _cell(dict(zip(source_headers, source.values, strict=False)), *names)


def extract_dates_2027(text: str) -> str:
    plain = html.unescape(re.sub(r"\s+", " ", re.sub("<[^<]+?>", " ", text))).strip()
    month = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    patterns = [
        rf"{month}\s+\d{{1,2}}\s*[-–]\s*\d{{1,2}},?\s*2027",
        rf"{month}\s+\d{{1,2}},?\s*2027\s*[-–]\s*{month}\s+\d{{1,2}},?\s*2027",
        rf"\d{{1,2}}\s*[-–]\s*\d{{1,2}}\s+{month}\s+2027",
        rf"\d{{1,2}}\s+{month}\s+2027\s*[-–]\s*\d{{1,2}}\s+{month}\s+2027",
    ]
    for pattern in patterns:
        match = re.search(pattern, plain, flags=re.I)
        if match:
            return match.group(0).strip()
    idx = plain.find("2027")
    if idx >= 0:
        start = max(0, idx - 60)
        end = min(len(plain), idx + 80)
        return plain[start:end].strip()
    return ""


def extract_city_country(text: str) -> tuple[str, str]:
    plain = html.unescape(re.sub(r"\s+", " ", re.sub("<[^<]+?>", " ", text))).strip()
    state_country = re.search(r"\bin\s+([A-Z][A-Za-z .'-]+),\s*([A-Z]{2}),\s*([A-Z][A-Za-z .'-]+)", plain)
    if state_country:
        country = re.sub(r"\s+(on|from|during|for).*$", "", state_country.group(3).strip()).strip()
        return f"{state_country.group(1).strip()}, {state_country.group(2).strip()}", country
    patterns = [
        r"\bin\s+([A-Z][A-Za-z .'-]+),\s*([A-Z][A-Za-z .'-]+)",
        r"\bLocation[:\s]+([A-Z][A-Za-z .'-]+),\s*([A-Z][A-Za-z .'-]+)",
        r"\bVenue[:\s]+([A-Z][A-Za-z .'-]+),\s*([A-Z][A-Za-z .'-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, plain)
        if match:
            city = match.group(1).strip()
            country = match.group(2).strip()
            city = re.sub(r"\s+(on|from|during|for)$", "", city).strip()
            country = re.sub(r"\s+(on|from|during|for).*$", "", country).strip()
            return city, country
    return "", ""


def build_compact_2027_row(
    source_headers: list[str],
    source: SourceRow,
    candidate: VerifiedCandidate | None,
) -> list[str]:
    evidence_text = candidate.evidence_snippet if candidate else ""
    city, country = extract_city_country(evidence_text)
    return [
        "FALSE",
        "2027",
        source.name,
        source_value(source_headers, source, "CORE23 Ranking", "CORE Ranking"),
        source_value(source_headers, source, "CORE link"),
        extract_dates_2027(evidence_text),
        city,
        country,
        candidate.site if candidate else "",
    ]


def row_already_mentions_year(source: SourceRow, year: int) -> bool:
    year_text = str(year)
    return any(year_text in value for value in source.values if value)


def load_sheets_service(credentials_file: str | None):
    try:
        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise SystemExit("Install Google clients with: uv add google-api-python-client google-auth") from exc

    path = credentials_file or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not path:
        raise SystemExit("Set GOOGLE_APPLICATION_CREDENTIALS or pass --credentials-file.")
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    credentials = Credentials.from_service_account_file(path, scopes=scopes)
    return build("sheets", "v4", credentials=credentials, cache_discovery=False)


def read_values(service, spreadsheet_id: str, sheet_name: str) -> list[list[str]]:
    result = (
        service.spreadsheets()
        .values()
        .get(spreadsheetId=spreadsheet_id, range=f"{sheet_name}!A:ZZ")
        .execute()
    )
    return result.get("values", [])


def ensure_sheet_exists(service, spreadsheet_id: str, sheet_name: str) -> None:
    spreadsheet = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    titles = {sheet["properties"]["title"] for sheet in spreadsheet.get("sheets", [])}
    if sheet_name in titles:
        return
    service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={"requests": [{"addSheet": {"properties": {"title": sheet_name}}}]},
    ).execute()


def ensure_target_header(service, spreadsheet_id: str, target_sheet: str, headers: list[str]) -> None:
    ensure_sheet_exists(service, spreadsheet_id, target_sheet)
    rows = read_values(service, spreadsheet_id, target_sheet)
    if rows and rows[0] == headers:
        return
    if rows:
        raise SystemExit(
            f"Target sheet {target_sheet!r} already has a different header. "
            "Use an empty target sheet for this candidate schema."
        )
    service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range=f"{target_sheet}!A1",
        valueInputOption="RAW",
        body={"values": [headers]},
    ).execute()


def write_values(service, spreadsheet_id: str, target_sheet: str, rows: list[list[str]]) -> None:
    if not rows:
        return
    service.spreadsheets().values().clear(
        spreadsheetId=spreadsheet_id,
        range=f"{target_sheet}!A:ZZ",
        body={},
    ).execute()
    service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range=f"{target_sheet}!A1",
        valueInputOption="RAW",
        body={"values": rows},
    ).execute()


def set_human_verified_checkboxes(
    service,
    spreadsheet_id: str,
    target_sheet: str,
    column_index: int,
    start_row: int,
    end_row: int,
) -> None:
    spreadsheet = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    sheet_id = None
    for sheet in spreadsheet.get("sheets", []):
        if sheet["properties"]["title"] == target_sheet:
            sheet_id = sheet["properties"]["sheetId"]
            break
    if sheet_id is None:
        return
    service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={
            "requests": [
                {
                    "setDataValidation": {
                        "range": {
                            "sheetId": sheet_id,
                            "startRowIndex": start_row - 1,
                            "endRowIndex": end_row,
                            "startColumnIndex": column_index,
                            "endColumnIndex": column_index + 1,
                        },
                        "rule": {
                            "condition": {"type": "BOOLEAN"},
                            "strict": True,
                            "showCustomUi": True,
                        },
                    }
                }
            ]
        },
    ).execute()


def require_search_provider(env: dict[str, str], allow_fallback: bool) -> str:
    if env.get("BRIGHTDATA_API_KEY"):
        return "brightdata"
    if env.get("SERPAPI_API_KEY"):
        return "serpapi"
    if allow_fallback:
        return "fallback"
    raise SystemExit(
        "Google SERP search is required for this run. Set BRIGHTDATA_API_KEY or SERPAPI_API_KEY in .env, "
        "or pass --allow-search-fallback for debugging only."
    )


def search_web(query: str, limit: int, timeout: int, provider: str) -> list[SearchResult]:
    try:
        if provider == "brightdata":
            return search_brightdata(
                query,
                limit,
                os.environ["BRIGHTDATA_API_KEY"],
                os.environ.get("BRIGHTDATA_ZONE", "serp_api1"),
                timeout,
            )
        if provider == "serpapi":
            return search_serpapi(
                query,
                limit,
                os.environ["SERPAPI_API_KEY"],
                timeout,
            )
        return search_duckduckgo(query, limit, timeout)
    except Exception:
        return []


def search_brightdata(query: str, limit: int, api_key: str, zone: str, timeout: int) -> list[SearchResult]:
    google_url = "https://www.google.com/search?" + urllib.parse.urlencode({"q": query})
    body = json.dumps(
        {
            "zone": zone,
            "url": google_url,
            "format": "json",
            "data_format": "parsed_light",
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://api.brightdata.com/request",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return parse_brightdata_results(data, limit)


def parse_brightdata_results(data: dict, limit: int) -> list[SearchResult]:
    body = data.get("body")
    if isinstance(body, str):
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            pass
    elif isinstance(body, dict):
        data = body
    raw_results = data.get("organic") or data.get("organic_results") or data.get("results") or []
    results: list[SearchResult] = []
    for item in raw_results:
        url = item.get("url") or item.get("link")
        title = item.get("title", "")
        snippet = item.get("description") or item.get("snippet", "")
        if url and urllib.parse.urlparse(url).scheme in {"http", "https"}:
            results.append(SearchResult(url=url, title=title, snippet=snippet))
        if len(results) >= limit:
            break
    return results


def search_serpapi(query: str, limit: int, api_key: str, timeout: int) -> list[SearchResult]:
    url = "https://serpapi.com/search.json?" + urllib.parse.urlencode(
        {
            "engine": "google",
            "api_key": api_key,
            "q": query,
            "num": min(limit, 10),
            "hl": "en",
        }
    )
    with urllib.request.urlopen(url, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return parse_serpapi_results(data, limit)


def parse_serpapi_results(data: dict, limit: int) -> list[SearchResult]:
    return [
        SearchResult(url=item.get("link", ""), title=item.get("title", ""), snippet=item.get("snippet", ""))
        for item in data.get("organic_results", [])
        if item.get("link")
    ][:limit]


def search_duckduckgo(query: str, limit: int, timeout: int) -> list[SearchResult]:
    url = "https://duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        text = response.read().decode("utf-8", errors="replace")
    results: list[SearchResult] = []
    pattern = re.compile(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.S)
    for href, title_html in pattern.findall(text):
        parsed_href = html.unescape(href)
        if "uddg=" in parsed_href:
            parsed = urllib.parse.urlparse(parsed_href)
            parsed_href = urllib.parse.parse_qs(parsed.query).get("uddg", [parsed_href])[0]
        title = html.unescape(re.sub("<[^<]+?>", "", title_html)).strip()
        if parsed_href and title:
            results.append(SearchResult(url=parsed_href, title=title))
        if len(results) >= limit:
            break
    return results


def fetch_page(url: str, timeout: int) -> tuple[int, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("content-type", "")
            if "text" not in content_type and "html" not in content_type and "xml" not in content_type:
                return response.status, ""
            return response.status, response.read(300_000).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, ""
    except Exception:
        return 0, ""


def name_tokens(name: str) -> list[str]:
    tokens = re.findall(r"[A-Za-zА-Яа-я0-9]{3,}", name.lower())
    skip = {
        "the",
        "and",
        "for",
        "with",
        "conference",
        "international",
        "workshop",
        "symposium",
        "global",
        "smart",
        "cities",
        "city",
        "summit",
        "urban",
    }
    return [token for token in tokens if token not in skip]


def name_acronyms(name: str) -> list[str]:
    acronyms = []
    for part in re.findall(r"\(([^)]*)\)", name):
        for token in re.findall(r"[A-Z]{2,}", part):
            if not token.startswith("20"):
                acronyms.append(token.lower())
    for token in re.findall(r"\b[A-Z][A-Z0-9]{2,}\b", name):
        if not token.startswith("20"):
            acronyms.append(token.lower())
    return list(dict.fromkeys(acronyms))


def name_matches_page(name: str, haystack: str) -> bool:
    lowered = haystack.lower()
    acronyms = name_acronyms(name)
    if acronyms:
        return any(acronym in lowered for acronym in acronyms)
    tokens = name_tokens(name)
    if len(tokens) <= 1:
        return bool(tokens and tokens[0] in lowered)
    matches = sum(1 for token in tokens[:10] if token in lowered)
    return matches >= 2


def result_has_conflicting_year(result_text: str, target_year: int) -> bool:
    years = set(re.findall(r"\b20\d{2}\b", result_text))
    return bool(years and str(target_year) not in years)


def candidate_from_verified_page(
    source: SourceRow,
    result: SearchResult,
    target_year: int,
    status: int,
    text: str,
) -> VerifiedCandidate | None:
    if status >= 400 or status == 0:
        return None
    page_text = text[:50_000].lower()
    if str(target_year) not in page_text:
        return None
    result_haystack = " ".join([result.title, result.snippet, result.url]).lower()
    if result_has_conflicting_year(result_haystack, target_year):
        return None
    if not name_matches_page(source.name, result_haystack):
        return None
    snippet = clean_snippet(text, target_year) or result.snippet
    return VerifiedCandidate(
        source=source,
        year=target_year,
        site=result.url,
        evidence_url=result.url,
        evidence_title=result.title,
        evidence_snippet=snippet[:500],
        confidence="high",
        notes="Verified by opening result page and matching target year/name token.",
    )


def clean_snippet(text: str, target_year: int) -> str:
    plain = html.unescape(re.sub(r"\s+", " ", re.sub("<[^<]+?>", " ", text))).strip()
    idx = plain.lower().find(str(target_year))
    if idx < 0:
        return plain[:300]
    start = max(0, idx - 140)
    end = min(len(plain), idx + 260)
    return plain[start:end]


def find_candidate(
    source: SourceRow,
    target_year: int,
    search_limit: int,
    search_timeout: int,
    page_timeout: int,
    search_provider: str,
) -> VerifiedCandidate | None:
    query = f'"{source.name}" {target_year} conference'
    for result in search_web(query, search_limit, search_timeout, search_provider):
        status, text = fetch_page(result.url, page_timeout)
        candidate = candidate_from_verified_page(source, result, target_year, status, text)
        if candidate:
            return candidate
    return None


def run(args: argparse.Namespace) -> int:
    run_id, checked_at = utc_stamp()
    years = sorted(set(args.years or [args.year]))
    search_provider = require_search_provider(os.environ, args.allow_search_fallback)
    service = load_sheets_service(args.credentials_file)
    source_values = read_values(service, args.spreadsheet_id, args.source_sheet)
    source_headers, source_rows = parse_source_rows(source_values, args.limit)
    if args.compact_2027:
        return run_compact_2027(args, service, source_headers, source_rows, search_provider)
    review_headers = build_review_headers(source_headers, years)

    output_rows: list[list[str]] = [review_headers]
    dry_run_payload = []
    for index, source in enumerate(source_rows, start=1):
        print(f"[{index}/{len(source_rows)}] {source.name}", file=sys.stderr, flush=True)
        candidates_by_year: dict[int, VerifiedCandidate | None] = {}
        statuses_by_year: dict[int, str] = {}
        for year in years:
            if row_already_mentions_year(source, year):
                candidates_by_year[year] = None
                statuses_by_year[year] = "already_has_year"
                continue
            candidate = find_candidate(
                source,
                year,
                args.search_limit,
                args.search_timeout,
                args.page_timeout,
                search_provider,
            )
            candidates_by_year[year] = candidate
            statuses_by_year[year] = "verified" if candidate else "no_verified_candidate"
        row = build_review_row(source_headers, source, candidates_by_year, statuses_by_year, years, run_id, checked_at)
        output_rows.append(row)
        dry_run_payload.append(
            {
                "source_row": source.row_number,
                "name": source.name,
                "years": {
                    str(year): {
                        "status": statuses_by_year[year],
                        "site": candidates_by_year[year].site if candidates_by_year[year] else "",
                    }
                    for year in years
                },
            }
        )

    if args.dry_run:
        print(json.dumps(dry_run_payload, ensure_ascii=False, indent=2))
        return 0

    ensure_sheet_exists(service, args.spreadsheet_id, args.target_sheet)
    write_values(service, args.spreadsheet_id, args.target_sheet, output_rows)
    checkbox_idx = review_headers.index("Human verified")
    set_human_verified_checkboxes(
        service,
        args.spreadsheet_id,
        args.target_sheet,
        checkbox_idx,
        start_row=2,
        end_row=len(output_rows),
    )
    print(f"Wrote {len(output_rows) - 1} copied review rows to {args.target_sheet}; run_id={run_id}")
    return 0


def run_compact_2027(args: argparse.Namespace, service, source_headers: list[str], source_rows: list[SourceRow], search_provider: str) -> int:
    target_year = 2027
    headers = build_compact_2027_headers()
    output_rows = [headers]
    dry_run_payload = []
    for index, source in enumerate(source_rows, start=1):
        print(f"[{index}/{len(source_rows)}] {source.name}", file=sys.stderr, flush=True)
        if row_already_mentions_year(source, target_year):
            candidate = None
        else:
            candidate = find_candidate(
                source,
                target_year,
                args.search_limit,
                args.search_timeout,
                args.page_timeout,
                search_provider,
            )
        output_rows.append(build_compact_2027_row(source_headers, source, candidate))
        dry_run_payload.append(
            {
                "source_row": source.row_number,
                "name": source.name,
                "site": candidate.site if candidate else "",
            }
        )
    if args.dry_run:
        print(json.dumps(dry_run_payload, ensure_ascii=False, indent=2))
        return 0
    ensure_sheet_exists(service, args.spreadsheet_id, args.target_sheet)
    write_values(service, args.spreadsheet_id, args.target_sheet, output_rows)
    set_human_verified_checkboxes(
        service,
        args.spreadsheet_id,
        args.target_sheet,
        column_index=0,
        start_row=2,
        end_row=len(output_rows),
    )
    print(f"Wrote {len(output_rows) - 1} compact 2027 rows to {args.target_sheet}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spreadsheet-id", required=True)
    parser.add_argument("--source-sheet", default="Conferences")
    parser.add_argument("--target-sheet", default="agent_candidates")
    parser.add_argument("--year", type=int, default=2027)
    parser.add_argument("--years", type=int, nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--search-limit", type=int, default=2)
    parser.add_argument("--search-timeout", type=int, default=6)
    parser.add_argument("--page-timeout", type=int, default=6)
    parser.add_argument("--allow-search-fallback", action="store_true")
    parser.add_argument("--compact-2027", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--credentials-file")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
