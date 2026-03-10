"""Reusable UI rendering components for the Streamlit frontend."""

from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

OKABE_ITO_PALETTE = [
    "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00",
    "#CC79A7", "#000000",
]


def inject_css() -> None:
    st.markdown(
        """
<style>
.synth-badge {
    background: #E69F00;
    color: white;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: bold;
    display: inline-block;
}
.cache-badge {
    background: #009E73;
    color: white;
    padding: 1px 6px;
    border-radius: 3px;
    font-size: 11px;
    display: inline-block;
}
.meta-info {
    color: #888;
    font-size: 12px;
}
.warn-box {
    background: #fff3cd;
    border-left: 4px solid #E69F00;
    padding: 8px 12px;
    border-radius: 4px;
    margin-bottom: 8px;
}
</style>
""",
        unsafe_allow_html=True,
    )


def render_geo_map(
    map_points: List[Dict[str, Any]],
) -> None:
    if not map_points:
        st.info("No location data to display.")
        return

    specialty_set = list(
        {p.get("specialty", "Unknown") for p in map_points}
    )
    color_lookup = {
        spec: OKABE_ITO_PALETTE[i % len(OKABE_ITO_PALETTE)]
        for i, spec in enumerate(specialty_set)
    }

    fig = go.Figure()
    for spec in specialty_set:
        subset = [
            p for p in map_points
            if p.get("specialty") == spec
        ]
        hover_texts = [
            (
                f"<b>{p.get('name', 'Unknown')}</b><br>"
                f"NPI: {p.get('npi', 'N/A')}<br>"
                f"Specialty: {spec}"
            )
            for p in subset
        ]
        fig.add_trace(
            go.Scattergeo(
                lat=[p.get("lat", 0) for p in subset],
                lon=[p.get("long", 0) for p in subset],
                text=hover_texts,
                hoverinfo="text",
                mode="markers",
                name=spec,
                marker=go.scattergeo.Marker(
                    size=10,
                    color=color_lookup[spec],
                    opacity=0.85,
                    line=go.scattergeo.marker.Line(
                        width=1,
                        color="white",
                    ),
                ),
            )
        )

    fig.update_layout(
        geo=go.layout.Geo(
            scope="usa",
            projection=go.layout.geo.Projection(
                type="albers usa"
            ),
            showland=True,
            landcolor="rgb(238, 238, 238)",
            showlakes=True,
            lakecolor="rgb(195, 218, 255)",
            showcoastlines=True,
            coastlinecolor="rgb(170, 170, 170)",
        ),
        height=340,
        margin=go.layout.Margin(l=0, r=0, t=10, b=0),
        legend=go.layout.Legend(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_sources_table(
    sources: List[Dict[str, Any]],
) -> None:
    if not sources:
        st.info("No sources returned.")
        return

    records = []
    for src in sources:
        snippet = src.get("snippet", "")
        truncated = (
            snippet[:100] + "..."
            if len(snippet) > 100
            else snippet
        )
        records.append(
            {
                "Provider": src.get("provider_name", ""),
                "NPI": src.get("npi", ""),
                "Snippet": truncated,
                "Relevance": round(
                    float(src.get("relevance_score", 0.0)), 3
                ),
            }
        )

    st.dataframe(
        pd.DataFrame(records),
        use_container_width=True,
        hide_index=True,
    )


def render_suggestion_buttons(
    suggestions: List[str],
    key_prefix: str,
) -> None:
    if not suggestions:
        return

    st.caption("Follow-up suggestions:")
    n_cols = min(len(suggestions), 3)
    columns = st.columns(n_cols)
    for idx, suggestion in enumerate(suggestions[:3]):
        label = (
            suggestion[:50] + "..."
            if len(suggestion) > 50
            else suggestion
        )
        with columns[idx % n_cols]:
            btn_key = f"{key_prefix}_sug_{idx}_{hash(suggestion)}"
            if st.button(
                label,
                key=btn_key,
                use_container_width=True,
            ):
                st.session_state["_prefill"] = suggestion
                st.rerun()


def render_response_header(
    cache_hit: bool,
    latency_ms: int,
    model_used: str,
) -> None:
    parts = []
    if cache_hit:
        parts.append(
            '<span class="cache-badge">⚡ Cached</span>'
        )
    if latency_ms:
        parts.append(
            f'<span class="meta-info">{latency_ms}ms'
            f" · {model_used}</span>"
        )
    if parts:
        st.markdown(
            " &nbsp; ".join(parts),
            unsafe_allow_html=True,
        )


def render_hallucination_warning() -> None:
    st.markdown(
        '<div class="warn-box">'
        "⚠️ <strong>Verification notice:</strong> "
        "Some citations in this response could not be fully "
        "verified against retrieved sources. "
        "Please review carefully."
        "</div>",
        unsafe_allow_html=True,
    )
