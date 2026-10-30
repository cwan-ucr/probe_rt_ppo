"""
Parse GRIDSMART traffic volume xls file and output averaged Thursday 15-min volumes.

Input:  data/GRIDSMART_TrafficVo_MayJune_15min.xls
        (one sheet per day, format: Volume_MMDDYYYY)
Output: outputs/calibration/gridsmart_thursday_avg.csv
        (96 rows x 15-min intervals x 4 directions, averaged across Thursdays)

Each sheet structure:
  Row 0-13: metadata (intersection name, date, daily totals)
  Row 14:   column headers (Northbound col 2, Eastbound col 5, Southbound col 7, Westbound col 8)
  Row 15:   first data row (00:00)
  Row 15-110: 96 data rows (00:00 to 23:45)
"""

import os
import sys
import datetime as dt
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(SRC_DIR)

XLS_FILE = os.path.join(ROOT_DIR, "data", "GRIDSMART_TrafficVo_MayJune_15min.xls")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs", "calibration")

# GRIDSMART column indices (0-based)
COL_TIME = 0
COL_NB = 2
COL_EB = 5
COL_SB = 7
COL_WB = 8

HEADER_ROW = 14  # row with "Northbound", "Eastbound", etc.
DATA_START = 15  # first actual 15-min data row
N_INTERVALS = 96  # 24 hours × 4 intervals


def sheet_name_to_date(name):
    """Parse sheet name 'Volume_MMDDYYYY' → datetime.date.

    Format has no leading zeros for month when month is single digit:
        Volume_5202021  → 5/20/2021
        Volume_612021   → 6/1/2021
        Volume_6272021  → 6/27/2021
    """
    if not name.startswith("Volume_"):
        return None
    ds = name[len("Volume_"):]  # e.g. "5202021" or "612021"

    # Last 4 chars = year
    if len(ds) < 5:
        return None
    year = int(ds[-4:])
    rest = ds[:-4]  # e.g. "520" or "61"

    # rest can be: MD (2), MDD or MMD (3), MMDD (4)
    # Strategy: try all valid splits, return the first valid date
    candidates = []
    n = len(rest)
    if n == 2:
        # 1-digit month, 1-digit day (e.g. "61" = June 1)
        candidates.append((int(rest[0]), int(rest[1])))
    elif n == 3:
        # 1-digit month, 2-digit day OR 2-digit month, 1-digit day
        candidates.append((int(rest[0]), int(rest[1:])))     # MDD
        candidates.append((int(rest[:2]), int(rest[2])))     # MMD
    elif n == 4:
        candidates.append((int(rest[:2]), int(rest[2:])))    # MMDD

    for month, day in candidates:
        try:
            return dt.date(year, month, day)
        except ValueError:
            continue
    return None


def read_sheet(xls, sheet_name):
    """Read one daily sheet and return 96-row DataFrame with directional flows.

    Returns: DataFrame with columns [time_hhmm, NB, EB, SB, WB], 96 rows
    """
    df_raw = pd.read_excel(xls, sheet_name, header=None)

    rows = []
    for i in range(DATA_START, DATA_START + N_INTERVALS):
        if i >= len(df_raw):
            break
        time_val = df_raw.iloc[i, COL_TIME]
        # time_val may be a string "HH:MM" or a datetime.time object
        if isinstance(time_val, dt.time):
            time_hhmm = time_val.strftime("%H:%M")
        elif pd.isna(time_val):
            continue
        else:
            time_hhmm = str(time_val).strip()

        def safe_num(x):
            if pd.isna(x):
                return 0
            try:
                return int(x)
            except (TypeError, ValueError):
                return 0

        rows.append({
            "time_hhmm": time_hhmm,
            "NB": safe_num(df_raw.iloc[i, COL_NB]),
            "EB": safe_num(df_raw.iloc[i, COL_EB]),
            "SB": safe_num(df_raw.iloc[i, COL_SB]),
            "WB": safe_num(df_raw.iloc[i, COL_WB]),
        })

    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("  GRIDSMART Thursday Volume Averager")
    print("=" * 60)

    if not os.path.exists(XLS_FILE):
        print(f"ERROR: GRIDSMART file not found: {XLS_FILE}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    xls = pd.ExcelFile(XLS_FILE)
    all_sheets = xls.sheet_names
    print(f"\nTotal sheets: {len(all_sheets)}")

    # Identify Thursdays
    thursday_sheets = []
    for name in all_sheets:
        d = sheet_name_to_date(name)
        if d is None:
            print(f"  WARN: unrecognized sheet name: {name}")
            continue
        if d.weekday() == 3:  # 0=Mon, 3=Thu
            thursday_sheets.append((name, d))

    print(f"\nIdentified {len(thursday_sheets)} Thursdays:")
    for name, d in thursday_sheets:
        print(f"  {d.strftime('%Y-%m-%d (%A)')}  sheet={name}")

    if len(thursday_sheets) == 0:
        print("ERROR: No Thursday sheets found")
        sys.exit(1)

    # Read each Thursday
    all_thursdays = []
    for name, d in thursday_sheets:
        print(f"\nReading {name} ({d})...")
        df = read_sheet(xls, name)
        df["date"] = d
        print(f"  Rows: {len(df)}, daily totals: "
              f"NB={df['NB'].sum()}, EB={df['EB'].sum()}, "
              f"SB={df['SB'].sum()}, WB={df['WB'].sum()}")
        all_thursdays.append(df)

    # Concatenate and compute mean + std per time_hhmm
    combined = pd.concat(all_thursdays, ignore_index=True)

    agg = combined.groupby("time_hhmm").agg(
        NB_avg=("NB", "mean"),
        EB_avg=("EB", "mean"),
        SB_avg=("SB", "mean"),
        WB_avg=("WB", "mean"),
        NB_std=("NB", "std"),
        EB_std=("EB", "std"),
        SB_std=("SB", "std"),
        WB_std=("WB", "std"),
        n_days=("NB", "count"),
    ).reset_index()

    # Sort by time
    agg = agg.sort_values("time_hhmm").reset_index(drop=True)

    # Round averages to integers (vehicle counts)
    for c in ["NB_avg", "EB_avg", "SB_avg", "WB_avg"]:
        agg[c] = agg[c].round().astype(int)

    # Compute coefficient of variation (CV = std/mean) per direction
    print("\n--- Variation Across Thursdays (CV = std / mean, daily) ---")
    daily_totals = []
    for df in all_thursdays:
        daily_totals.append({
            "date": df["date"].iloc[0],
            "NB": df["NB"].sum(),
            "EB": df["EB"].sum(),
            "SB": df["SB"].sum(),
            "WB": df["WB"].sum(),
        })
    dt_df = pd.DataFrame(daily_totals)
    print(dt_df.to_string(index=False))
    print()
    for d in ["NB", "EB", "SB", "WB"]:
        cv = dt_df[d].std() / dt_df[d].mean() if dt_df[d].mean() > 0 else 0
        flag = "⚠️" if cv > 0.3 else "✓"
        print(f"  {d}: mean={dt_df[d].mean():.0f}, std={dt_df[d].std():.0f}, CV={cv:.3f} {flag}")

    # Save
    out_path = os.path.join(OUTPUT_DIR, "gridsmart_thursday_avg.csv")
    agg.to_csv(out_path, index=False)
    print(f"\n✓ Saved: {out_path}")
    print(f"  {len(agg)} rows × {len(agg.columns)} columns")

    # Also save daily totals for reference
    totals_path = os.path.join(OUTPUT_DIR, "gridsmart_thursday_daily_totals.csv")
    dt_df.to_csv(totals_path, index=False)
    print(f"✓ Saved: {totals_path}")

    # Print sample rows around peak hours
    print("\nSample averaged rows (morning peak):")
    morning = agg[agg["time_hhmm"].str.startswith(("07:", "08:", "09:"))]
    print(morning[["time_hhmm", "NB_avg", "EB_avg", "SB_avg", "WB_avg"]].to_string(index=False))


if __name__ == "__main__":
    main()
