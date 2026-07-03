# 4:3 Arctic Sankey Panels

## Scope

Re-render these four Russian May–October Arctic Sankey diagrams as 4:3 landscape images:

- `nao_health_may_oct_ru_4x3.png`
- `yakut_chuk_port_may_oct_ru_4x3.png`
- `mezen_port_may_oct_ru_4x3.png`
- `yanao_kras_airport_may_oct_ru_4x3.png`

## Design

Render each selected diagram directly at `1200 × 900` logical pixels using the existing
`create_clean_sankey` data and styling path. Do not crop, pad, or resize the existing
PNGs. With Plotly scale 2, each output is `2400 × 1800` physical pixels. Leave the
other region/service diagrams unchanged.

Display the no-provider label as two lines (`НЕТ` / `ПОСТАВЩИКА`) only on the
rightmost no-provider node. Keep all intermediate no-provider nodes and flows but
leave their labels blank. Left-align and shift the `Потребители` header so it does
not overlap the May header.

Also create a `2 × 2` contact sheet from the four new 4:3 files so their layout can
be reviewed together. The sheet is `4800 × 3600` pixels and preserves each panel
without distortion.

## Verification

- Confirm all four individual PNGs are exactly `2400 × 1800` pixels.
- Confirm the contact sheet contains the intended four diagrams.
- Open the contact sheet and inspect label readability, visible flows, and clipping.
