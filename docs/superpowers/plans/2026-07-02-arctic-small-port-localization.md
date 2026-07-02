# Arctic Small-Port Localization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace visible marina labels with «малый порт» and regenerate the third seasonal-network scheme fully in Russian.

**Architecture:** Keep internal service keys stable, change only display-label mappings and thesis prose. Add one deterministic Pillow renderer for the third scheme and update the Excalidraw text source to match.

**Tech Stack:** Python, Pillow, Excalidraw JSON

---

### Task 1: Update visible terminology

**Files:**
- Modify: `arctic_access/scripts/plotter/plotter_flow_sankey.py`
- Modify: `arctic_access/scripts/plotter/plotter_multilayer_service_network.py`
- Modify: `itmo-phd-thesis-template-en/Dissertation/chapter4.tex`

- [ ] Change display mappings and prose to «малый порт» while preserving the `marina` data key.
- [ ] Search source files to verify no old visible term remains.

### Task 2: Regenerate the Russian third scheme

**Files:**
- Modify: `arctic_access/plots/excildraw/exildraw_arctic_paper_imgs.excalidraw`
- Create: `arctic_access/scripts/plotter/render_seasonal_network_scheme_cold_ru.py`
- Modify: `itmo-phd-thesis-template-en/images/ch4/arctic/seasonal_network_scheme_cold.png`

- [ ] Translate Excalidraw text while retaining formula symbols.
- [ ] Render a Russian PNG matching the existing dark hand-drawn scheme.
- [ ] Compile the renderer, verify image dimensions and inspect the PNG directly.
