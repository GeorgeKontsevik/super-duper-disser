# Four-city heat experiment composite

## Goal

Produce one publication-ready figure for Gothenburg, Hrodna, Graz, and Innsbruck that matches the supplied 4-by-2 reference while remaining readable at document scale.

## Layout

- Four city columns in this order: Gothenburg, Hrodna, Graz, Innsbruck.
- Top row: pedestrian UTCI links with water and buildings.
- Bottom row: residential-building increase in travel time for heat versus baseline.
- One shared legend for the complete top row and one shared legend for the complete bottom row. Per-panel legends are omitted.
- City/panel titles and both legends use substantially larger type than the current composite, targeting approximately 2 to 2.5 times the apparent size in the supplied image.

## Implementation

Extend the existing heat-story rendering path in `scripts/render_debrecen_heat_story_maps.py` instead of reproducing map logic. Render all eight axes in one Matplotlib figure so titles, spacing, legends, and typography are controlled natively and remain sharp. Keep the existing colors, layer ordering, UTCI classes, and delta-time bins.

Write a single high-resolution PNG under the existing heat experiment output tree. Do not overwrite the individual city maps.

## Verification

- Confirm all four city datasets load and all eight panels contain plotted features.
- Inspect the final PNG directly for title/legend readability, unclipped text, consistent extents, visible water/building/road layers, and correct city order.
- Confirm exactly two legends are present and no per-panel legends remain.
