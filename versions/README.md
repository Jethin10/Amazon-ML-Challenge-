# Version routes

Each folder records one release route and points to its canonical implementation.
Shared Python modules remain under `src/` so imports and reproducibility stay
stable. Generated data remains outside Git except for the final V8 leaderboard
TSV.

| Version | Purpose | Folder |
| --- | --- | --- |
| V1 | Baseline matching | `v1/` |
| V2 | Improved features and audit | `v2/` |
| V3 | Expanded candidate features | `v3/` |
| V4 | Stronger LightGBM baseline | `v4/` |
| V5 | Full scalable candidate pipeline | `v5/` |
| V6 | Selected pair transformer reranking | `v6/` |
| V7 | Full retained candidate reranking | `v7/` |
| V8 | Final exclusivity and France recovery | `v8/` |
