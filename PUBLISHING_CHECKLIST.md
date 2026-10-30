# Checks before publishing this staged bundle

This folder is a local handoff package, not a claim that every included input
may be redistributed publicly. Review these items before copying it into the
GitHub repository.

1. **Commercial probe-derived inputs:** the 32 files under
   `data/geojson-1-15-min-interval/processed/` contain TomTom-derived speed,
   travel-time, and sample-size statistics, not just segment geometry. Confirm
   that the applicable agreement permits redistribution. If it does not, do
   **not** upload those files; instead provide an authorized-access procedure
   or release a separately validated non-proprietary replacement. Omitting or
   modifying them will change input hashes and may prevent exact replay of the
   archived checkpoints.
2. **Map and calibrated-demand rights:** check the terms for the derived OSM
   network, detector geometry, signal plans, and calibrated synthetic trip file.
   Add the required attribution and any applicable data license. Also review
   the GRIDSMART-derived count CSVs and calibration/optimization summaries.
   The original GRIDSMART workbook **is included in this local staging folder**
   for reproducibility; remove it from the public copy if its terms do not
   permit redistribution. Do not assume purchasing or processing a dataset
   grants the right to redistribute it.
3. **Diaz checkpoint gap:** the local `v11` archive contains Diaz training and
   evaluation CSVs but no Diaz final `.pt` or `.pt.meta.json`. Retrieve both
   from HPCC if you want others to rerun Diaz evaluation from a checkpoint.
   The included Diaz code/launcher is the restored near-stop-bar legacy
   configuration; archive version tags should be kept distinct.
4. **Environment:** `requirements.txt` lists packages but is not a lockfile.
   Export and test the actual Python 3.9/Conda environment used on HPCC for
   stronger version reproducibility. SUMO 1.20.0 must be installed separately.
5. **Licenses and citation:** choose explicit software and data licenses and
   add the final manuscript citation/DOI. Do not apply a permissive code
   license to third-party input data by implication.
6. **Final Git review:** after copying into the new repository, inspect
   `git status --short` and `git diff --cached --stat` before committing. Check
   that every intended CSV, JSON, and `.pt.meta.json` is present and that no
   private, raw commercial, or unrelated working files are staged.
7. **Upstream provenance:** `fixed_timing_60s.add.xml` was regenerated for this
   bundle and has three minor-program min-green warnings. The copied historical
   bandwidth result was not rerun against it. The original fringe/segment-flow
   intermediates were unavailable; the bundled versions were regenerated from
   current source inputs. The older approach inflow is preserved separately.
   Do not claim byte-identical end-to-end reproduction without reconciling
   these versions and the historical random seeds.
