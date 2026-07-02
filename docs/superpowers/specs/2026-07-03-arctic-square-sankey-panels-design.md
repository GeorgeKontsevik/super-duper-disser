# Square Arctic Sankey Panels

## Scope

Re-render these four Russian May–October Arctic Sankey diagrams as square images:

- `nao_health_may_oct_ru.png`
- `yakut_chuk_port_may_oct_ru.png`
- `mezen_port_may_oct_ru.png`
- `yanao_kras_airport_may_oct_ru.png`

## Design

Render each selected diagram directly at `1000 × 1000` pixels using the existing
`create_clean_sankey` data and styling path. Do not crop, pad, or resize the existing
wide PNGs. Leave the other region/service diagrams unchanged.

Also create a `2 × 2` contact sheet from the four new square files so their layout can
be reviewed together. Preserve one square cell per diagram without distortion.

## Verification

- Confirm all four individual PNGs are exactly `1000 × 1000` pixels.
- Confirm the contact sheet contains the intended four diagrams.
- Open the contact sheet and inspect label readability, visible flows, and clipping.
