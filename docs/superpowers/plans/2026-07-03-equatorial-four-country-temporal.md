# Equatorial Four-Country Temporal Figure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render COL, LBR, CMR, and GAB rainfall/delay time series as a readable square 2×2 figure.

**Architecture:** Add a focused CSV-backed renderer. Filter the saved weekly country mechanism table to the four selected countries, apply shared rainfall and delay limits, and use one common legend.

**Tech Stack:** Python, pandas, Matplotlib

---

### Task 1: Render the square four-country figure

**Files:**
- Create: `equatorial/scripts/render_four_country_temporal_ru.py`
- Create: `itmo-phd-thesis-template-en/images/ch4/temporal_rain_burden_four_countries_square_ru.png`

- [ ] Verify the 2400×2400 target is absent.
- [ ] Read `weekly_country_mechanism.csv` and assert 53 rows for each selected country.
- [ ] Render a shared-axis 2×2 figure with Russian labels and one legend.
- [ ] Verify dimensions, panel count, country selection, and direct PNG appearance.
- [ ] Run `py_compile` and `git diff --check`.
