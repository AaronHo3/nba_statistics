from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="NBA Historical Dashboards", layout="wide")

DATA_DIR = Path("data/raw")

PLAYER_CANONICAL_COLUMNS = {
    "season": ["season", "year", "yr", "seas_id"],
    "player": ["player", "player_name", "name"],
    "team": ["tm", "team", "team_abbr", "franch"],
    "position": ["pos", "position"],
    "minutes": ["mp", "minutes", "min", "mp_per_game", "mp_per_36_min"],
    "points": ["pts", "points", "pts_per_game", "pts_per_36_min", "pts_per_100_poss"],
    "assists": ["ast", "assists", "ast_per_game", "ast_per_36_min", "ast_per_100_poss"],
    "rebounds": ["trb", "reb", "rebounds", "trb_per_game", "trb_per_36_min", "trb_per_100_poss"],
    "steals": ["stl", "steals", "stl_per_game", "stl_per_36_min", "stl_per_100_poss"],
    "blocks": ["blk", "blocks", "blk_per_game", "blk_per_36_min", "blk_per_100_poss"],
}
PLAYER_NUMERIC_COLUMNS = ["minutes", "points", "assists", "rebounds", "steals", "blocks"]
REQUIRED_PLAYER_COLUMNS = {"season", "player", "team", "position", "minutes", "points", "assists"}



def _season_sort_key(value: str) -> int:
    match = re.search(r"(\d{4})", str(value))
    return int(match.group(1)) if match else -1


def _decade_from_year(year: int) -> str | None:
    if year < 0:
        return None
    decade_start = (year // 10) * 10
    return f"{decade_start}s"



def _find_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    by_lower = {c.lower(): c for c in df.columns}
    for alias in aliases:
        if alias in by_lower:
            return by_lower[alias]
    return None



def _canonicalize_player(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rename_map: dict[str, str] = {}
    missing: list[str] = []
    for canonical_name, aliases in PLAYER_CANONICAL_COLUMNS.items():
        matched = _find_column(df, aliases)
        if matched:
            rename_map[matched] = canonical_name
        else:
            missing.append(canonical_name)

    out = df.rename(columns=rename_map).copy()

    for col in PLAYER_NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "season" in out.columns:
        out["season"] = out["season"].astype(str)
    if "player" in out.columns:
        out["player"] = out["player"].astype(str)
    for col in ("team", "position"):
        if col in out.columns:
            out[col] = out[col].astype(str)

    return out, missing



def _collapse_traded_rows(df: pd.DataFrame) -> pd.DataFrame:
    needed = {"player", "season", "team"}
    if not needed.issubset(df.columns):
        return df

    def _pick(group: pd.DataFrame) -> pd.DataFrame:
        tot = group[group["team"].str.upper() == "TOT"]
        if not tot.empty:
            return tot.iloc[[0]]
        if len(group) == 1:
            return group

        out = group.iloc[[0]].copy()
        for col in PLAYER_NUMERIC_COLUMNS:
            if col in group.columns:
                out[col] = group[col].sum(skipna=True)
        out["team"] = "MULTI"
        return out

    return df.groupby(["player", "season"], group_keys=False, dropna=False).apply(_pick).reset_index(drop=True)



@st.cache_data(show_spinner=False)
def load_player_data_from_csv(path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path)
    canonical_df, missing = _canonicalize_player(df)
    canonical_df = _collapse_traded_rows(canonical_df)
    return canonical_df, missing



def load_csv_options(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    csvs = sorted(directory.glob("*.csv"))
    if csvs:
        preferred = [p for p in csvs if "player" in p.name.lower()]
        return preferred + [p for p in csvs if p not in preferred]
    return []



@st.cache_data(show_spinner=False)
def inspect_player_csv(path: Path) -> tuple[list[str], list[str]]:
    sample = pd.read_csv(path, nrows=1)
    canonical, _ = _canonicalize_player(sample)
    columns = list(canonical.columns)
    missing_required = sorted(REQUIRED_PLAYER_COLUMNS - set(columns))
    return columns, missing_required



def _pca_2d(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] == 0:
        return np.zeros((0, 2))
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    if vt.shape[0] == 1:
        comp = centered @ vt[:1].T
        return np.hstack([comp, np.zeros((comp.shape[0], 1))])
    return centered @ vt[:2].T


def _build_player_feature_table(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feature_candidates = ["points", "assists", "rebounds", "steals", "blocks", "minutes"]
    feature_cols = [c for c in feature_candidates if c in df.columns]

    agg_map: dict[str, tuple[str, str]] = {
        "seasons": ("season", "nunique"),
        "career_minutes": ("minutes", "sum"),
    }
    for col in feature_cols:
        agg_map[col] = (col, "mean")

    group = df.groupby("player", as_index=False).agg(**agg_map)
    for col in feature_cols + ["seasons", "career_minutes"]:
        if col in group.columns:
            group[col] = pd.to_numeric(group[col], errors="coerce")
    group = group.dropna(subset=feature_cols)
    return group, feature_cols


@st.cache_data(show_spinner=False)
def load_team_data(summary_path: Path, totals_path: Path) -> pd.DataFrame:
    summaries = pd.read_csv(summary_path)
    totals = pd.read_csv(totals_path)

    summaries.columns = [c.lower() for c in summaries.columns]
    totals.columns = [c.lower() for c in totals.columns]

    required_summary = {"season", "team", "lg", "w", "l", "o_rtg", "pace"}
    required_totals = {"season", "team", "lg", "x2p", "x3p", "ft", "pts"}
    if not required_summary.issubset(summaries.columns):
        raise ValueError(f"Team Summaries is missing columns: {sorted(required_summary - set(summaries.columns))}")
    if not required_totals.issubset(totals.columns):
        raise ValueError(f"Team Totals is missing columns: {sorted(required_totals - set(totals.columns))}")

    # Keep NBA only for team evolution visuals.
    summaries = summaries[summaries["lg"].astype(str).str.upper() == "NBA"].copy()
    totals = totals[totals["lg"].astype(str).str.upper() == "NBA"].copy()

    keep_summary = [
        c
        for c in ["season", "team", "lg", "abbreviation", "w", "l", "o_rtg", "d_rtg", "n_rtg", "pace"]
        if c in summaries.columns
    ]
    keep_totals = [c for c in ["season", "team", "lg", "g", "pts", "x2p", "x3p", "x3pa", "ft", "ast", "fg"] if c in totals.columns]

    summaries = summaries[keep_summary].copy()
    totals = totals[keep_totals].copy()

    merged = summaries.merge(totals, on=["season", "team", "lg"], how="left")

    merged["season"] = merged["season"].astype(str)
    merged["team"] = merged["team"].astype(str)

    for col in ["w", "l", "o_rtg", "d_rtg", "n_rtg", "pace", "g", "pts", "x2p", "x3p", "x3pa", "ft", "ast", "fg"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    # NBA introduced the 3-point line in 1979-80; treat pre-1980 missing 3PT stats as zero.
    season_year = pd.to_numeric(merged["season"], errors="coerce")
    pre_1980 = season_year < 1980
    if "x3p" in merged.columns:
        merged.loc[pre_1980 & merged["x3p"].isna(), "x3p"] = 0.0
    if "x3pa" in merged.columns:
        merged.loc[pre_1980 & merged["x3pa"].isna(), "x3pa"] = 0.0

    games = (merged["w"].fillna(0) + merged["l"].fillna(0)).replace(0, pd.NA)
    merged["win_pct"] = merged["w"] / games
    merged["pts_2p"] = merged["x2p"] * 2
    merged["pts_3p"] = merged["x3p"] * 3
    merged["pts_ft"] = merged["ft"]

    if "g" in merged.columns:
        g_nonzero = merged["g"].replace(0, pd.NA)
        merged["pts_per_game"] = merged["pts"] / g_nonzero
        if "x3pa" in merged.columns:
            merged["x3pa_per_game"] = merged["x3pa"] / g_nonzero

    if {"ast", "fg"}.issubset(merged.columns):
        merged["ast_rate"] = merged["ast"] / merged["fg"].replace(0, pd.NA)

    return merged



def render_player_tab() -> None:
    st.subheader("Player Performance Explorer")
    st.caption("Career trend and skill profile for historical players.")

    csv_paths = load_csv_options(DATA_DIR)
    if not csv_paths:
        st.warning("No CSV files found in `data/raw`.")
        return

    schema_info = {p: inspect_player_csv(p) for p in csv_paths}
    compatible_paths = [p for p in csv_paths if not schema_info[p][1]]
    incompatible_paths = [p for p in csv_paths if schema_info[p][1]]

    if not compatible_paths:
        st.error("No compatible player CSV files found for this dashboard.")
        return

    default_idx = 0
    preferred_name = "Player Totals.csv"
    for i, p in enumerate(compatible_paths):
        if p.name == preferred_name:
            default_idx = i
            break

    selected_csv = st.selectbox(
        "Player Data Source",
        options=compatible_paths,
        index=default_idx,
        format_func=lambda p: p.name,
        key="player_csv_selector",
    )

    if incompatible_paths:
        with st.expander("Show incompatible CSV files"):
            for p in incompatible_paths:
                st.write(f"- {p.name} (missing: {', '.join(schema_info[p][1])})")

    df, missing_columns = load_player_data_from_csv(selected_csv)
    if not REQUIRED_PLAYER_COLUMNS.issubset(df.columns):
        st.error(f"Missing required canonical columns: {sorted(REQUIRED_PLAYER_COLUMNS - set(df.columns))}")
        return

    if missing_columns:
        st.info("Optional columns missing: " + ", ".join(sorted(set(missing_columns))))

    c1, c2, c3, c4 = st.columns([2, 2, 2, 3])

    seasons = sorted(df["season"].dropna().unique(), key=_season_sort_key)
    teams = sorted(df["team"].dropna().unique())
    positions = sorted(df["position"].dropna().unique())

    with c1:
        selected_seasons = st.multiselect(
            "Year (Season)", options=seasons, default=seasons[-5:] if len(seasons) >= 5 else seasons, key="p_seasons"
        )
    with c2:
        selected_teams = st.multiselect("Team", options=teams, default=teams, key="p_teams")
    with c3:
        selected_positions = st.multiselect("Position", options=positions, default=positions, key="p_positions")
    with c4:
        min_minutes = int(df["minutes"].min(skipna=True) if df["minutes"].notna().any() else 0)
        max_minutes = int(df["minutes"].max(skipna=True) if df["minutes"].notna().any() else 0)
        minute_threshold = st.slider(
            "Minutes played threshold",
            min_value=min(0, min_minutes),
            max_value=max(1, max_minutes),
            value=max(0, int(max_minutes * 0.25)) if max_minutes > 0 else 0,
            step=max(1, max_minutes // 100) if max_minutes > 100 else 1,
            key="p_minutes",
        )

    filtered = df.copy()
    if selected_seasons:
        filtered = filtered[filtered["season"].isin(selected_seasons)]
    if selected_teams:
        filtered = filtered[filtered["team"].isin(selected_teams)]
    if selected_positions:
        filtered = filtered[filtered["position"].isin(selected_positions)]
    filtered = filtered[filtered["minutes"].fillna(0) >= minute_threshold]

    if filtered.empty:
        st.warning("No player data matches the selected filters.")
        return

    st.markdown("### Player Career Chart")
    player_choices = sorted(filtered["player"].dropna().unique())
    default_player = "LeBron James" if "LeBron James" in player_choices else player_choices[0]
    if "selected_player" not in st.session_state or st.session_state["selected_player"] not in player_choices:
        st.session_state["selected_player"] = default_player

    p1, p2 = st.columns([2, 2])
    with p1:
        selected_player = st.selectbox(
            "Player",
            options=player_choices,
            index=player_choices.index(st.session_state["selected_player"]),
            key="p_selected_player",
        )
    st.session_state["selected_player"] = selected_player
    with p2:
        trajectory_metric = st.selectbox(
            "Career metric",
            options=[m for m in ["points", "assists", "rebounds", "minutes", "steals", "blocks"] if m in filtered.columns],
            key="p_metric",
        )

    career_df = df[df["player"] == selected_player].copy()
    career_df = career_df.sort_values("season", key=lambda s: s.map(_season_sort_key))
    career_clean = career_df.dropna(subset=[trajectory_metric]).copy()

    card1, card2, card3 = st.columns(3)
    card1.metric("Career Seasons", f"{career_df['season'].nunique():,}")
    card2.metric(f"Career Avg {trajectory_metric.title()}", f"{career_clean[trajectory_metric].mean():.2f}")
    card3.metric(f"Career Peak {trajectory_metric.title()}", f"{career_clean[trajectory_metric].max():.2f}")

    line = go.Figure()
    line.add_trace(
        go.Scatter(
            x=career_clean["season"],
            y=career_clean[trajectory_metric],
            mode="lines+markers",
            line=dict(color="#1264A3", width=3),
            marker=dict(size=7, color="#0B1F3A"),
            fill="tozeroy",
            fillcolor="rgba(18, 100, 163, 0.12)",
            hovertemplate="Season %{x}<br>" + trajectory_metric.title() + ": %{y:.2f}<extra></extra>",
        )
    )
    line.update_layout(
        height=430,
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
        title=f"{selected_player} - {trajectory_metric.title()} by Season",
        xaxis_title="Season",
        yaxis_title=trajectory_metric.title(),
    )
    st.plotly_chart(line, use_container_width=True)

    career_cols = [
        c for c in ["season", "team", "position", "minutes", "points", "assists", "rebounds", "steals", "blocks"] if c in career_df.columns
    ]
    if career_cols:
        st.markdown(f"**{selected_player} career stats (table)**")
        st.dataframe(
            career_df[career_cols].sort_values("season", key=lambda s: s.map(_season_sort_key)),
            use_container_width=True,
            hide_index=True,
        )

    if {"points", "assists", "rebounds", "steals", "blocks"}.issubset(df.columns):
        st.markdown("### Player Profile Radar")
        player_seasons = sorted(career_df["season"].dropna().unique(), key=_season_sort_key)
        radar_season = st.selectbox(
            "Season for radar profile",
            options=player_seasons,
            index=max(0, len(player_seasons) - 1),
            key="p_radar_season",
        )

        season_df = df[df["season"] == radar_season].copy()
        season_df_threshold = season_df[season_df["minutes"].fillna(0) >= minute_threshold].copy()
        if len(season_df_threshold) >= 20:
            season_df = season_df_threshold

        radar_metrics = ["points", "assists", "rebounds", "steals", "blocks"]
        metric_labels = ["PTS", "AST", "REB", "STL", "BLK"]

        player_row = season_df[season_df["player"] == selected_player]
        can_render_radar = True
        if player_row.empty:
            fallback_row = df[(df["season"] == radar_season) & (df["player"] == selected_player)]
            if fallback_row.empty:
                st.info("Selected player has no record for radar profile in this season.")
                can_render_radar = False
            else:
                player_row = fallback_row.iloc[[0]]

        if can_render_radar:
            percentiles = []
            for metric in radar_metrics:
                pool = season_df.dropna(subset=[metric]).copy()
                if pool.empty:
                    percentiles.append(0.0)
                    continue
                player_value = float(player_row[metric].iloc[0]) if pd.notna(player_row[metric].iloc[0]) else 0.0
                percentile = float((pool[metric] <= player_value).mean() * 100.0)
                percentiles.append(percentile)

            radar = go.Figure()
            radar.add_trace(
                go.Scatterpolar(
                    r=percentiles + [percentiles[0]],
                    theta=metric_labels + [metric_labels[0]],
                    fill="toself",
                    fillcolor="rgba(11, 31, 58, 0.25)",
                    line=dict(color="#0B1F3A", width=3),
                    marker=dict(color="#1264A3", size=7),
                    name=f"{selected_player} ({radar_season})",
                    hovertemplate="%{theta}: %{r:.1f} percentile<extra></extra>",
                )
            )
            radar.update_layout(
                template="plotly_white",
                polar=dict(
                    bgcolor="rgba(18, 100, 163, 0.05)",
                    radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%", gridcolor="rgba(0,0,0,0.12)"),
                    angularaxis=dict(gridcolor="rgba(0,0,0,0.08)"),
                ),
                showlegend=False,
                height=520,
                margin=dict(l=20, r=20, t=50, b=20),
                title="Skill Profile Percentiles vs Same-Season Peers",
            )
            st.plotly_chart(radar, use_container_width=True)



def render_team_tab() -> None:
    st.subheader("Team Evolution Dashboard")
    st.caption("How team identity changes across decades: wins, offense, scoring mix, and tempo-efficiency profile.")

    summaries_path = DATA_DIR / "Team Summaries.csv"
    totals_path = DATA_DIR / "Team Totals.csv"

    if not summaries_path.exists() or not totals_path.exists():
        st.error("Team dashboard needs `Team Summaries.csv` and `Team Totals.csv` in `data/raw`.")
        return

    try:
        team_df = load_team_data(summaries_path, totals_path)
    except ValueError as exc:
        st.error(str(exc))
        return

    team_df = team_df.dropna(subset=["season", "team", "win_pct", "o_rtg", "pace"]).copy()
    if team_df.empty:
        st.warning("No team records available after cleaning.")
        return

    years = sorted(team_df["season"].unique(), key=_season_sort_key)
    teams = sorted(team_df["team"].unique())

    f1, f2, f3 = st.columns([3, 3, 2])
    with f1:
        selected_years = st.multiselect(
            "Year (Season)", options=years, default=years[-20:] if len(years) >= 20 else years, key="t_years"
        )
    with f2:
        selected_teams = st.multiselect("Teams", options=teams, default=teams, key="t_teams")
    with f3:
        min_games = int((team_df["w"].fillna(0) + team_df["l"].fillna(0)).min())
        max_games = int((team_df["w"].fillna(0) + team_df["l"].fillna(0)).max())
        games_threshold = st.slider(
            "Games threshold",
            min_value=max(1, min_games),
            max_value=max(1, max_games),
            value=max(1, int(max_games * 0.75)),
            key="t_games",
        )

    filtered = team_df.copy()
    if selected_years:
        filtered = filtered[filtered["season"].isin(selected_years)]
    if selected_teams:
        filtered = filtered[filtered["team"].isin(selected_teams)]
    filtered = filtered[(filtered["w"].fillna(0) + filtered["l"].fillna(0)) >= games_threshold]

    if filtered.empty:
        st.warning("No team data matches the selected filters.")
        return

    selected_team_default = "Los Angeles Lakers" if "Los Angeles Lakers" in teams else teams[0]
    if "selected_team_evolution" not in st.session_state or st.session_state["selected_team_evolution"] not in teams:
        st.session_state["selected_team_evolution"] = selected_team_default

    st.markdown("### Pace vs Efficiency")
    st.caption("Each dot is one NBA team-season. X-axis is pace, Y-axis is offensive rating, and color shows win percentage.")
    scatter = go.Figure()
    scatter.add_trace(
        go.Scatter(
            x=filtered["pace"],
            y=filtered["o_rtg"],
            mode="markers",
            marker=dict(
                size=10,
                color=filtered["win_pct"],
                colorscale="Viridis",
                colorbar=dict(title="Win%"),
                line=dict(color="rgba(255,255,255,0.4)", width=0.8),
            ),
            customdata=filtered[["team", "season", "win_pct"]].values,
            hovertemplate="%{customdata[0]} (%{customdata[1]})<br>Pace: %{x:.1f}<br>Off Rtg: %{y:.1f}<br>Win%: %{customdata[2]:.3f}<extra></extra>",
        )
    )
    scatter.update_layout(
        height=420,
        template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Pace",
        yaxis_title="Offensive Rating",
    )

    st.plotly_chart(scatter, use_container_width=True)

    selected_team = st.selectbox(
        "Selected team",
        options=teams,
        index=teams.index(st.session_state["selected_team_evolution"]),
        key="t_selected_team",
    )
    st.session_state["selected_team_evolution"] = selected_team

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Team Win Percentage Over Time")
        history = team_df[team_df["team"] == selected_team].copy()
        if selected_years:
            history = history[history["season"].isin(selected_years)]
        history = history.sort_values("season", key=lambda s: s.map(_season_sort_key))

        league_avg = filtered.groupby("season", as_index=False)["win_pct"].mean()
        league_avg = league_avg.sort_values("season", key=lambda s: s.map(_season_sort_key))

        line = go.Figure()
        line.add_trace(
            go.Scatter(
                x=league_avg["season"],
                y=league_avg["win_pct"],
                mode="lines",
                line=dict(color="rgba(80,80,80,0.5)", width=2, dash="dash"),
                name="League Avg",
            )
        )
        line.add_trace(
            go.Scatter(
                x=history["season"],
                y=history["win_pct"],
                mode="lines+markers",
                line=dict(color="#0B1F3A", width=3),
                marker=dict(size=7, color="#1264A3"),
                fill="tozeroy",
                fillcolor="rgba(18, 100, 163, 0.12)",
                name=selected_team,
            )
        )
        line.update_layout(
            height=420,
            template="plotly_white",
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis_title="Win%",
            xaxis_title="Season",
            yaxis=dict(tickformat=".0%"),
            legend=dict(orientation="h", y=1.05, x=0),
        )
        st.plotly_chart(line, use_container_width=True)

    with c2:
        st.markdown("### Team Scoring Distribution")
        seasons_for_bar = sorted(filtered["season"].unique(), key=_season_sort_key)
        selected_season_bar = st.selectbox(
            "Season for scoring mix",
            options=seasons_for_bar,
            index=max(0, len(seasons_for_bar) - 1),
            key="t_scoring_season",
        )

        bar_df = filtered[filtered["season"] == selected_season_bar].copy()
        bar_df = bar_df.sort_values("pts", ascending=False).head(15)

        stacked = go.Figure()
        stacked.add_bar(x=bar_df["team"], y=bar_df["pts_2p"], name="2PT Points", marker_color="#1f77b4")
        stacked.add_bar(x=bar_df["team"], y=bar_df["pts_3p"], name="3PT Points", marker_color="#17becf")
        stacked.add_bar(x=bar_df["team"], y=bar_df["pts_ft"], name="FT Points", marker_color="#ff7f0e")
        stacked.update_layout(
            barmode="stack",
            height=420,
            template="plotly_white",
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Team",
            yaxis_title="Points",
            legend=dict(orientation="h", y=1.05, x=0),
        )
        st.plotly_chart(stacked, use_container_width=True)

    st.markdown("### Offensive Rating Heatmap")
    heat_df = filtered.pivot_table(index="team", columns="season", values="o_rtg", aggfunc="mean")
    heat_df = heat_df.reindex(sorted(heat_df.columns, key=_season_sort_key), axis=1)
    heat_df = heat_df.sort_index()

    heat = go.Figure(
        data=go.Heatmap(
            z=heat_df.values,
            x=heat_df.columns,
            y=heat_df.index,
            colorscale="YlGnBu",
            colorbar=dict(title="Off Rtg"),
            hovertemplate="Team: %{y}<br>Season: %{x}<br>Off Rtg: %{z:.2f}<extra></extra>",
        )
    )
    heat.update_layout(
        height=650,
        template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Season",
        yaxis_title="Team",
    )
    st.plotly_chart(heat, use_container_width=True)


def render_era_tab() -> None:
    st.subheader("Era Comparison Dashboard")
    st.caption("Compare style and scoring trends across NBA decades.")

    summaries_path = DATA_DIR / "Team Summaries.csv"
    totals_path = DATA_DIR / "Team Totals.csv"
    if not summaries_path.exists() or not totals_path.exists():
        st.error("Era dashboard needs `Team Summaries.csv` and `Team Totals.csv` in `data/raw`.")
        return

    try:
        df = load_team_data(summaries_path, totals_path).copy()
    except ValueError as exc:
        st.error(str(exc))
        return

    df["year"] = df["season"].map(_season_sort_key)
    df["era"] = df["year"].map(_decade_from_year)
    df = df[df["era"].notna()].copy()

    if df.empty:
        st.warning("No rows matched decade grouping.")
        return

    era_order = [f"{d}s" for d in sorted(df["year"].dropna().astype(int).floordiv(10).mul(10).unique())]

    c1, c2 = st.columns([2, 3])
    with c1:
        selected_eras = st.multiselect("Decades", options=era_order, default=era_order, key="era_selected")
    with c2:
        teams = sorted(df["team"].dropna().unique())
        selected_teams = st.multiselect("Optional team filter", options=teams, default=teams, key="era_teams")

    filtered = df.copy()
    if selected_eras:
        filtered = filtered[filtered["era"].isin(selected_eras)]
    if selected_teams:
        filtered = filtered[filtered["team"].isin(selected_teams)]

    if filtered.empty:
        st.warning("No era data matches selected filters.")
        return

    scoring = (
        filtered.groupby("era", as_index=False)[["pts_2p", "pts_3p", "pts_ft"]]
        .mean()
        .set_index("era")
        .reindex([e for e in era_order if e in filtered["era"].unique()])
        .reset_index()
    )

    totals = (scoring["pts_2p"] + scoring["pts_3p"] + scoring["pts_ft"]).replace(0, pd.NA)
    scoring["share_2p"] = scoring["pts_2p"] / totals
    scoring["share_3p"] = scoring["pts_3p"] / totals
    scoring["share_ft"] = scoring["pts_ft"] / totals

    assist = (
        filtered.groupby("era", as_index=False)["ast_rate"]
        .mean()
        .set_index("era")
        .reindex([e for e in era_order if e in filtered["era"].unique()])
        .reset_index()
    )
    threes = (
        filtered.groupby("era", as_index=False)["x3pa_per_game"]
        .mean()
        .set_index("era")
        .reindex([e for e in era_order if e in filtered["era"].unique()])
        .reset_index()
    )
    pace = (
        filtered.groupby("era", as_index=False)["pace"]
        .mean()
        .set_index("era")
        .reindex([e for e in era_order if e in filtered["era"].unique()])
        .reset_index()
    )

    row1, row2 = st.columns(2)

    with row1:
        st.markdown("### Scoring Distribution")
        stacked = go.Figure()
        stacked.add_bar(x=scoring["era"], y=scoring["share_2p"], name="2PT Share", marker_color="#1f77b4")
        stacked.add_bar(x=scoring["era"], y=scoring["share_3p"], name="3PT Share", marker_color="#17becf")
        stacked.add_bar(x=scoring["era"], y=scoring["share_ft"], name="FT Share", marker_color="#ff7f0e")
        stacked.update_layout(
            barmode="stack",
            template="plotly_white",
            height=400,
            yaxis=dict(tickformat=".0%"),
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", y=1.05, x=0),
        )
        st.plotly_chart(stacked, use_container_width=True)

    with row2:
        st.markdown("### Assist Rates")
        assist_bar = go.Figure(
            data=[
                go.Bar(
                    x=assist["era"],
                    y=assist["ast_rate"],
                    marker_color="#0B1F3A",
                    hovertemplate="Era: %{x}<br>Assist Rate: %{y:.3f}<extra></extra>",
                )
            ]
        )
        assist_bar.update_layout(
            template="plotly_white",
            height=400,
            yaxis=dict(tickformat=".0%"),
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(assist_bar, use_container_width=True)

    with row1:
        st.markdown("### 3-Point Attempts")
        threes_bar = go.Figure(
            data=[
                go.Bar(
                    x=threes["era"],
                    y=threes["x3pa_per_game"],
                    marker_color="#1264A3",
                    hovertemplate="Era: %{x}<br>3PA/G: %{y:.2f}<extra></extra>",
                )
            ]
        )
        threes_bar.update_layout(
            template="plotly_white",
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis_title="Average Team 3PA per Game",
        )
        st.plotly_chart(threes_bar, use_container_width=True)

    with row2:
        st.markdown("### Pace")
        pace_bar = go.Figure(
            data=[
                go.Bar(
                    x=pace["era"],
                    y=pace["pace"],
                    marker_color="#005B96",
                    hovertemplate="Era: %{x}<br>Pace: %{y:.2f}<extra></extra>",
                )
            ]
        )
        pace_bar.update_layout(
            template="plotly_white",
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis_title="Average Pace",
        )
        st.plotly_chart(pace_bar, use_container_width=True)


def render_similar_players_tab() -> None:
    st.subheader("Find Similar Players")
    st.caption("Select a player and discover statistically similar profiles.")

    csv_paths = load_csv_options(DATA_DIR)
    if not csv_paths:
        st.warning("No CSV files found in `data/raw`.")
        return

    schema_info = {p: inspect_player_csv(p) for p in csv_paths}
    compatible_paths = [p for p in csv_paths if not schema_info[p][1]]
    if not compatible_paths:
        st.error("No compatible player CSV files found.")
        return

    default_idx = 0
    for i, p in enumerate(compatible_paths):
        if p.name == "Player Totals.csv":
            default_idx = i
            break

    selected_csv = st.selectbox(
        "Player Data Source",
        options=compatible_paths,
        index=default_idx,
        format_func=lambda p: p.name,
        key="sim_player_csv_selector",
    )

    df, _ = load_player_data_from_csv(selected_csv)
    if not REQUIRED_PLAYER_COLUMNS.issubset(df.columns):
        st.error(f"Missing required canonical columns: {sorted(REQUIRED_PLAYER_COLUMNS - set(df.columns))}")
        return

    seasons = sorted(df["season"].dropna().unique(), key=_season_sort_key)
    c1, c2, c3 = st.columns([3, 2, 2])
    with c1:
        selected_seasons = st.multiselect(
            "Seasons to include",
            options=seasons,
            default=seasons[-10:] if len(seasons) >= 10 else seasons,
            key="sim_seasons",
        )
    with c2:
        top_k = st.slider("Number of similar players", min_value=3, max_value=12, value=6, key="sim_top_k")
    with c3:
        min_seasons = st.slider("Min seasons played", min_value=1, max_value=10, value=3, key="sim_min_seasons")

    filtered = df.copy()
    if selected_seasons:
        filtered = filtered[filtered["season"].isin(selected_seasons)]

    features_df, feature_cols = _build_player_feature_table(filtered)
    if len(feature_cols) < 3:
        st.warning("Not enough numeric player features available in this CSV for similarity matching.")
        return

    features_df = features_df[features_df["seasons"] >= min_seasons].copy()
    features_df = features_df.sort_values("player").reset_index(drop=True)
    if len(features_df) < top_k + 1:
        st.warning("Not enough players after filters. Expand season range or reduce constraints.")
        return

    player_choices = features_df["player"].tolist()
    default_player = "Stephen Curry" if "Stephen Curry" in player_choices else player_choices[0]
    selected_player = st.selectbox(
        "Player",
        options=player_choices,
        index=player_choices.index(default_player),
        key="sim_selected_player",
    )

    method = st.selectbox(
        "Similarity method",
        options=["Interpretable Weighted Distance", "Cosine Similarity (legacy)"],
        index=0,
        key="sim_method",
        help="Interpretable mode uses weighted standardized stat differences and exposes per-stat contributions.",
    )

    x = features_df[feature_cols].to_numpy(dtype=float)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    z = (x - mean) / std
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = z / norms

    selected_idx = int(features_df.index[features_df["player"] == selected_player][0])
    contribution_df = pd.DataFrame(index=features_df.index, columns=feature_cols, data=0.0)

    if method == "Interpretable Weighted Distance":
        weights = np.full(len(feature_cols), 1.0 / len(feature_cols))
        abs_diff = np.abs(z - z[selected_idx])
        weighted_diff = abs_diff * weights
        distance = weighted_diff.sum(axis=1)
        similarity = np.exp(-distance)
        similarity[selected_idx] = 1.0

        contribution_share = np.divide(
            weighted_diff,
            np.where(distance.reshape(-1, 1) == 0, 1.0, distance.reshape(-1, 1)),
        )
        contribution_df = pd.DataFrame(contribution_share, columns=feature_cols, index=features_df.index)

        pairwise_distance = np.abs(z[:, None, :] - z[None, :, :]) * weights.reshape(1, 1, -1)
        pairwise_distance = pairwise_distance.sum(axis=2)
        pair_similarity = np.exp(-pairwise_distance)
    else:
        similarity = unit @ unit[selected_idx]
        similarity[selected_idx] = 1.0
        pair_similarity = unit @ unit.T

    features_df["similarity"] = similarity

    similar_df = (
        features_df[features_df["player"] != selected_player]
        .sort_values("similarity", ascending=False)
        .head(top_k)
        .copy()
    )

    st.markdown("### Selected Player Baseline")
    baseline_cols = ["player", "seasons", "career_minutes"] + feature_cols
    baseline = features_df.loc[[selected_idx], baseline_cols].copy()
    st.dataframe(baseline, use_container_width=True, hide_index=True)

    st.markdown("### Similar Players")
    table_cols = ["player", "similarity"] + feature_cols
    table_df = similar_df[table_cols].copy()
    table_df["similarity"] = table_df["similarity"].map(lambda v: f"{v:.3f}")
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    if method == "Interpretable Weighted Distance":
        breakdown_rows: list[dict[str, object]] = []
        for _, row in similar_df.iterrows():
            idx = int(row.name)
            entry: dict[str, object] = {"player": row["player"], "similarity": f"{row['similarity']:.3f}"}
            for col in feature_cols:
                entry[f"{col}_delta"] = abs(float(features_df.loc[selected_idx, col]) - float(row[col]))
                entry[f"{col}_contrib_pct"] = float(contribution_df.loc[idx, col]) * 100.0
            breakdown_rows.append(entry)
        breakdown_df = pd.DataFrame(breakdown_rows)
        st.markdown("### Why These Players Are Similar")
        st.caption("`*_delta` shows raw stat gap from selected player. `*_contrib_pct` shows share of total similarity distance.")
        st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

    coords = _pca_2d(z)
    embed = features_df[["player", "similarity"]].copy()
    embed["pc1"] = coords[:, 0]
    embed["pc2"] = coords[:, 1]
    focus_players = [selected_player] + similar_df["player"].tolist()
    embed["is_focus"] = embed["player"].isin(focus_players)

    row1, row2 = st.columns(2)

    with row1:
        st.markdown("### Similarity Network")
        node_df = embed[embed["player"].isin(focus_players)].copy().reset_index(drop=True)
        pos = {r["player"]: (r["pc1"], r["pc2"]) for _, r in node_df.iterrows()}
        sim_map = dict(zip(features_df["player"], features_df["similarity"]))
        idx_map = dict(zip(features_df["player"], features_df.index))

        network = go.Figure()
        for player in similar_df["player"].tolist():
            x0, y0 = pos[selected_player]
            x1, y1 = pos[player]
            w = max(0.0, float(sim_map[player]))
            network.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(color="rgba(18,100,163,0.45)", width=1 + 5 * w),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        others = similar_df["player"].tolist()
        for i, p1 in enumerate(others):
            for p2 in others[i + 1 :]:
                w = float(pair_similarity[idx_map[p1], idx_map[p2]])
                threshold = 0.7 if method == "Interpretable Weighted Distance" else 0.9
                if w >= threshold:
                    x0, y0 = pos[p1]
                    x1, y1 = pos[p2]
                    network.add_trace(
                        go.Scatter(
                            x=[x0, x1],
                            y=[y0, y1],
                            mode="lines",
                            line=dict(color="rgba(11,31,58,0.2)", width=1 + 3 * w),
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )

        node_sizes = [28 if p == selected_player else 14 + 10 * max(0.0, float(sim_map[p])) for p in focus_players]
        node_colors = ["#0B1F3A" if p == selected_player else "#1264A3" for p in focus_players]
        network.add_trace(
            go.Scatter(
                x=[pos[p][0] for p in focus_players],
                y=[pos[p][1] for p in focus_players],
                mode="markers+text",
                text=focus_players,
                textposition="top center",
                marker=dict(size=node_sizes, color=node_colors, line=dict(color="white", width=1)),
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            )
        )
        network.update_layout(
            template="plotly_white",
            height=520,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        st.plotly_chart(network, use_container_width=True)

    with row2:
        st.markdown("### Scatter Embedding")
        bg_trace = go.Scatter(
            x=embed["pc1"],
            y=embed["pc2"],
            mode="markers",
            marker=dict(size=6, color="rgba(120,120,120,0.22)"),
            text=embed["player"],
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
        focus_df = embed[embed["is_focus"]].copy()
        focus_trace = go.Scatter(
            x=focus_df["pc1"],
            y=focus_df["pc2"],
            mode="markers+text",
            text=focus_df["player"],
            textposition="top center",
            marker=dict(
                size=[22 if p == selected_player else 12 for p in focus_df["player"]],
                color=["#0B1F3A" if p == selected_player else "#1264A3" for p in focus_df["player"]],
                line=dict(color="white", width=1),
            ),
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
        embedding = go.Figure(data=[bg_trace, focus_trace])
        embedding.update_layout(
            template="plotly_white",
            height=520,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Embedding Axis 1",
            yaxis_title="Embedding Axis 2",
        )
        st.plotly_chart(embedding, use_container_width=True)


st.title("NBA Historical Dashboards")
st.caption("Four views: player performance, team evolution, era comparison, and similar-player discovery.")

player_tab, team_tab, era_tab, similar_tab = st.tabs(
    ["Player Performance Explorer", "Team Evolution Dashboard", "Era Comparison Dashboard", "Find Similar Players"]
)
with player_tab:
    render_player_tab()
with team_tab:
    render_team_tab()
with era_tab:
    render_era_tab()
with similar_tab:
    render_similar_players_tab()
