"""
Data processing module for flow-based demand calibration.

Pipeline:
1. parse_gridsmart.py        — Parse GRIDSMART xls, average 5 Thursdays
2. extract_iowa_tomtom.py    — Extract Iowa approach last-segment data
3. compute_penetration.py    — Compute probe vehicle penetration rate
4. estimate_segment_flows.py — Estimate per-segment volumes for all 29 approaches
5. find_corridor_boundary_edges.py — Identify SUMO boundary edges (hybrid)
6. compute_fringe_flows.py   — Aggregate approach flows to fringe inflows
"""
