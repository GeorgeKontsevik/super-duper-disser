import unittest

from scripts.conference_tracker_agent import (
    REVIEW_HEADERS,
    SearchResult,
    SourceRow,
    VerifiedCandidate,
    build_review_headers,
    build_review_row,
    build_compact_2027_headers,
    build_compact_2027_row,
    candidate_from_verified_page,
    require_search_provider,
    normalize_source_row,
    row_already_mentions_year,
    parse_brightdata_results,
    name_matches_page,
)


class ConferenceTrackerAgentTest(unittest.TestCase):
    def test_normalize_source_row_preserves_original_values(self):
        headers = [
            "Column 1",
            "CORE23 Ranking",
            "Dates / Location source",
            "Журнал публикации (потенциальный)",
            "Квартиль (потенциальный)",
            "Submission DDL",
            "Workshop link",
        ]
        values = [
            "WSDM",
            "A",
            "https://wsdm-conference.org/2026/",
            "Information Processing and Management",
            "Q1",
            "7 August, 2025",
            "Urban Workshop",
        ]

        row = normalize_source_row(headers, values, row_number=3)

        self.assertEqual(row.row_number, 3)
        self.assertEqual(row.name, "WSDM")
        self.assertEqual(row.current_site, "https://wsdm-conference.org/2026/")
        self.assertEqual(row.values[: len(values)], values)

    def test_candidate_from_verified_page_rejects_404_and_missing_year(self):
        source = SourceRow(row_number=2, headers=["Column 1"], values=["ICUI"], name="ICUI")
        result = SearchResult(url="https://www.isocui.org/icui2027/", title="ICUI 2027")

        self.assertIsNone(
            candidate_from_verified_page(source, result, target_year=2027, status=404, text="ICUI 2027")
        )
        self.assertIsNone(
            candidate_from_verified_page(source, result, target_year=2027, status=200, text="ICUI 2025")
        )

    def test_candidate_from_verified_page_accepts_page_with_year_and_name_token(self):
        source = SourceRow(row_number=2, headers=["Column 1"], values=["ICUI"], name="ICUI")
        result = SearchResult(url="https://www.isocui.org/icui2027/", title="ICUI 2027")

        candidate = candidate_from_verified_page(
            source,
            result,
            target_year=2027,
            status=200,
            text="The International Conference on Urban Informatics ICUI 2027 call for papers.",
        )

        self.assertEqual(candidate.site, "https://www.isocui.org/icui2027/")
        self.assertEqual(candidate.confidence, "high")

    def test_candidate_from_verified_page_rejects_directory_result_without_name_in_result(self):
        source = SourceRow(
            row_number=2,
            headers=["Column 1"],
            values=["Global Smart Cities Summit cum The 4th International Conference on Urban Informatics (GSCS & ICUI 2025)"],
            name="Global Smart Cities Summit cum The 4th International Conference on Urban Informatics (GSCS & ICUI 2025)",
        )
        result = SearchResult(
            url="https://www.webmobi.com/discovery/events/in/hong-kong/topics/ai-for-future-cities/2025",
            title="AI for Future Cities Events",
            snippet="Events in Hong Kong.",
        )

        self.assertIsNone(
            candidate_from_verified_page(
                source,
                result,
                target_year=2027,
                status=200,
                text="ICUI 2027 appears somewhere in a generic directory page.",
            )
        )

    def test_candidate_from_verified_page_rejects_result_with_conflicting_year(self):
        source = SourceRow(row_number=2, headers=["Column 1"], values=["ICUI"], name="ICUI")
        result = SearchResult(
            url="https://www.polyu.edu.hk/scri/news-and-events/event/2025/08-05-2025/",
            title="ICUI 2025",
            snippet="Urban informatics conference.",
        )

        self.assertIsNone(
            candidate_from_verified_page(
                source,
                result,
                target_year=2026,
                status=200,
                text="ICUI 2026 appears somewhere else on the page.",
            )
        )

    def test_build_compact_2027_row_has_only_requested_columns(self):
        headers = ["Column 1", "CORE23 Ranking", "CORE link"]
        source = SourceRow(
            row_number=3,
            headers=headers,
            values=["WSDM", "A*", "https://portal.core.edu.au/conf-ranks/1/"],
            name="WSDM",
        )
        candidate = VerifiedCandidate(
            source=source,
            year=2027,
            site="https://www.wsdm-conference.org/2027/",
            evidence_url="https://www.wsdm-conference.org/2027/",
            evidence_title="WSDM 2027",
            evidence_snippet="WSDM 2027 will be held in Washington, DC, United States on February 22-26, 2027.",
            confidence="high",
            notes="Verified.",
        )

        self.assertEqual(
            build_compact_2027_headers(),
            [
                "Human verified",
                "Year",
                "Conference",
                "CORE23 Ranking",
                "CORE link",
                "Dates 2027",
                "City",
                "Country",
                "Verified site",
            ],
        )
        self.assertEqual(
            build_compact_2027_row(headers, source, candidate),
            [
                "FALSE",
                "2027",
                "WSDM",
                "A*",
                "https://portal.core.edu.au/conf-ranks/1/",
                "February 22-26, 2027",
                "Washington, DC",
                "United States",
                "https://www.wsdm-conference.org/2027/",
            ],
        )

    def test_name_matches_page_requires_acronym_when_name_has_acronym(self):
        name = "Global Smart Cities Summit cum The 4th International Conference on Urban Informatics (GSCS & ICUI 2025)"

        self.assertFalse(name_matches_page(name, "AI for future cities 2027"))
        self.assertTrue(name_matches_page(name, "ICUI 2027 call for papers"))

    def test_build_review_row_uses_source_headers_plus_review_fields(self):
        headers = ["Column 1", "Dates / Location source", "Main Conference Website", "Submission DDL"]
        source = SourceRow(
            row_number=3,
            headers=headers,
            values=["WSDM", "https://wsdm-conference.org/2026/", "", "7 August, 2025"],
            name="WSDM",
            current_site="https://wsdm-conference.org/2026/",
        )
        candidate = VerifiedCandidate(
            source=source,
            year=2027,
            site="https://www.wsdm-conference.org/2027/",
            evidence_url="https://www.wsdm-conference.org/2027/",
            evidence_title="WSDM 2027",
            evidence_snippet="WSDM 2027 conference site",
            confidence="high",
            notes="Verified by page content.",
        )

        row = build_review_row(
            headers,
            source,
            {2026: None, 2027: candidate},
            {2026: "already_has_year", 2027: "verified"},
            [2026, 2027],
            run_id="run-a",
            checked_at="2026-07-07T09:00:00Z",
        )

        self.assertEqual(
            build_review_headers(headers, [2026, 2027]),
            headers
            + REVIEW_HEADERS
            + [
                "agent_status_2026",
                "agent_candidate_site_2026",
                "agent_confidence_2026",
                "agent_evidence_url_2026",
                "agent_evidence_title_2026",
                "agent_evidence_snippet_2026",
                "agent_notes_2026",
                "agent_status_2027",
                "agent_candidate_site_2027",
                "agent_confidence_2027",
                "agent_evidence_url_2027",
                "agent_evidence_title_2027",
                "agent_evidence_snippet_2027",
                "agent_notes_2027",
            ],
        )
        self.assertEqual(row[:4], ["WSDM", "https://wsdm-conference.org/2026/", "", "7 August, 2025"])
        self.assertEqual(row[4:8], ["FALSE", "run-a", "2026-07-07T09:00:00Z", "3"])
        self.assertEqual(row[8:15], ["already_has_year", "", "", "", "", "", ""])
        self.assertEqual(
            row[15:],
            [
                "verified",
                "https://www.wsdm-conference.org/2027/",
                "high",
                "https://www.wsdm-conference.org/2027/",
                "WSDM 2027",
                "WSDM 2027 conference site",
                "Verified by page content.",
            ],
        )

    def test_row_already_mentions_year(self):
        row = SourceRow(
            row_number=3,
            headers=["Column 1", "Conference dates 2026"],
            values=["WSDM", "February 2026"],
            name="WSDM",
        )

        self.assertTrue(row_already_mentions_year(row, 2026))
        self.assertFalse(row_already_mentions_year(row, 2027))

    def test_require_search_provider_fails_without_serp_key_unless_fallback_allowed(self):
        env = {"BRIGHTDATA_API_KEY": "", "SERPAPI_API_KEY": ""}

        with self.assertRaises(SystemExit):
            require_search_provider(env, allow_fallback=False)

        self.assertEqual(require_search_provider(env, allow_fallback=True), "fallback")
        self.assertEqual(
            require_search_provider({"BRIGHTDATA_API_KEY": "key"}, allow_fallback=False),
            "brightdata",
        )
        self.assertEqual(
            require_search_provider({"SERPAPI_API_KEY": "key"}, allow_fallback=False),
            "serpapi",
        )

    def test_parse_brightdata_results_uses_organic_results(self):
        data = {
            "organic": [
                {
                    "url": "https://wsdm-conference.org/2027/",
                    "title": "WSDM 2027",
                    "description": "The official WSDM 2027 conference website.",
                }
            ]
        }

        results = parse_brightdata_results(data, limit=5)

        self.assertEqual(results, [SearchResult("https://wsdm-conference.org/2027/", "WSDM 2027", "The official WSDM 2027 conference website.")])


if __name__ == "__main__":
    unittest.main()
