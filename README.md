# NBA Historical Dashboards

Interactive Streamlit app with four dashboards:
- Player Performance Explorer
- Team Evolution Dashboard
- Era Comparison Dashboard
- Find Similar Players

## Quick Start (Make)

```bash
make setup
make run
```

Run `make help` to see all commands.

## 1) Setup (Manual)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Get Data

Download the Kaggle dataset:

- https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats

Option A (recommended): one command download

```bash
make download-data
```

Option B: manual download and place CSV files in:

- `data/raw/`

For `make download-data`, ensure Kaggle credentials are configured (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` + `KAGGLE_KEY`).

Recommended file types are per-season player stats tables (for example files with columns similar to `Player`, `Season`/`Year`, `Tm`, `Pos`, `MP`, `PTS`, `AST`, `TRB`, `STL`, `BLK`).

## 3) Run

```bash
streamlit run app.py
```

## Useful Make commands

```bash
make help          # list all targets
make setup         # create .venv + install deps
make download-data # fetch Kaggle dataset and stage CSVs in data/raw
make validate-data # ensure CSV files exist in data/raw
make list-data     # show CSV files in data/raw
make run           # run Streamlit app
make run-dev       # run Streamlit with autosave reload
make eda           # quick terminal dataset profile
make notebook      # open Jupyter Notebook
make check         # syntax-check app.py
make clean         # remove local caches
```

## Included dashboard sections

- Player tab
- Filters: Season, Team, Position, Minutes played threshold
- Player Career chart: styled trajectory over seasons
- Player Profile Radar: PTS/AST/REB/STL/BLK percentile vs same-season peers
- Career stats table for selected player
- Team tab
- Line chart: Team win percentage over time (selected team + league average)
- Stacked bar: Team scoring distribution (2PT/3PT/FT points)
- Heatmap: Offensive rating by team/year
- Scatter: Pace vs efficiency
- Era tab
- Scoring distribution by decade
- Assist rates by decade
- 3-point attempts by decade
- Pace by decade
- Similar players tab
- Similar-player ranking table
- Selected-player baseline stats table
- Interpretable similarity mode (weighted standardized stat differences)
- Per-stat contribution breakdown (`delta` + contribution %)
- Similarity network graph
- 2D scatter embedding of player profiles

## Notes

- If a player has multiple team rows in a season, `TOT` is preferred when present.
- If no `TOT` row exists for a multi-team season, numeric stats are aggregated and team is labeled `MULTI`.
