from scripts import render_heat_service_city_pairs_and_routes as styles


def test_shared_heat_route_style_contract() -> None:
    assert styles.HOME_COLOR == "#16a34a"
    assert styles.SERVICE_COLOR == "#2563eb"
    assert styles.BASELINE_PT_COLOR == "#7f1d1d"
    assert styles.HEAT_PT_COLOR == "#16a34a"
    assert styles.HOME_MARKER == "o"
    assert styles.SERVICE_MARKER == "*"
    assert styles.HOME_MARKER_SIZE >= 100
    assert styles.SERVICE_MARKER_SIZE >= 100
    assert styles.PT_LINESTYLE == (0, (2, 1.5))
