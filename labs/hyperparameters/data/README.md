# Leaf photographs

`ex08_leaves_images.zip` is an unchanged copy of the MLR teaching dataset:
https://github.com/tomasvicar/MLR-public/blob/master/exercises/data/ex08_leaves_images.zip

SHA-256: `f7bbaa9aa161bd1c334e2fe3fe8b5e93af216ff1664f2571d3572eace555cf53`.

It contains 820 JPEG photographs at 96 × 96 pixels: apple, cherry, chestnut
and maple, with 175 training and 30 held-out images per class. This lab splits
the 120 held-out images into 60 validation and 60 test examples with seed 0,
and resizes images to 48 × 48 during loading. The archive itself is unchanged.

Provenance: MLR `exercises/ex08/sources.md`, section
`exercises/data/ex08_leaves_images.zip`; preparation code is
`exercises/data/ex08_prepare_leaves.py` in the source MLR repository.
The original photographs were taken by MLR students in previous years;
the MLR preparation script selects balanced subsets, crops to a central square
and resizes the photographs. Borrowing MLR teaching material was agreed with
Roman Jakubíček on 1 September 2026, as recorded in AUI's `AGENTS.md`.
No separate open-content licence is asserted for these photographs.

The author requested this repository copy on 21 September 2026 to remove
the notebook's runtime dependency on the MLR repository. The archive is
published with the student notebook; extracted `leaves/` files are a cache.
