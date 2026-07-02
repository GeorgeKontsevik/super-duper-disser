# Heat route map style

## Scope

Apply one consistent route style to all heat-experiment maps that show homes, destination services, and baseline/heat routes. Reuse the existing renderers and their context-loading paths; do not introduce a theme framework.

## Visual design

- Draw the selected home as a large green star.
- Draw destination services as large blue stars.
- Draw baseline PT segments in burgundy with a frequent dash pattern.
- Draw heat-aware PT segments in green with the same frequent dash pattern.
- Keep walking segments solid and preserve their baseline/heat distinction.
- Increase the visibility of background street/network, building, and water layers slightly while keeping routes and endpoint markers dominant.
- Update every affected legend to match the plotted markers, colors, and line styles.

## Implementation

Change existing route-rendering functions and shared style values where the current code already provides a common path. Avoid parallel helpers or new dependencies. Cover the style contract with focused tests where practical, then regenerate all available heat route-map outputs.

## Verification

- Run focused tests and compile the touched Python modules.
- Inspect generated PNG dimensions and non-empty output files.
- Open representative outputs from each affected renderer and visually confirm marker identity, PT dash patterns/colors, context visibility, and legend accuracy.
