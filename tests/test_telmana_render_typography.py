from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/render_telmana_connector_visual_matrix_v2.py"


def test_telmana_figures_use_large_arial_typography():
    source = SCRIPT.read_text()

    assert '"font.family": "Arial"' in source
    assert "CARD_FONT_SIZE = 15" in source
    assert "LEGEND_FONT_SIZE = 13" in source
    assert "TITLE_FONT_SIZE = 14" in source


def test_telmana_figures_use_plain_top_left_titles():
    source = SCRIPT.read_text()

    assert "title_letter" not in source
    assert '"а) Исходные слои"' not in source
    assert '"СПб, квартал Тельмана:' not in source
    assert '0.012,\n        0.985,\n        title,\n        ha="left"' in source
    assert '0.012,\n        0.985,\n        "Исходные слои",\n        ha="left"' in source


def test_context_layers_are_stacked_vertically():
    source = SCRIPT.read_text()

    assert "figsize=(4.2, 15.5)" in source
    assert "nrows=5,\n        ncols=1" in source
    assert "context_fig.add_subplot(context_gs[i, 0])" in source
