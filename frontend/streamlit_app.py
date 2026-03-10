"""Main Streamlit application entry point."""

import asyncio
import concurrent.futures
import json
import uuid
from typing import Any, Dict, List, Optional

import httpx
import streamlit as st

from frontend.ui_components import (
    inject_css,
    render_geo_map,
    render_hallucination_warning,
    render_response_header,
    render_sources_table,
    render_suggestion_buttons,
)

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Healthcare Provider Search",
    page_icon=":hospital:",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

_SPECIALTY_OPTIONS = [
    "",
    "Family Medicine",
    "Internal Medicine",
    "Pediatrics",
    "Cardiology",
    "Psychiatry",
    "Obstetrics & Gynecology",
    "Orthopedic Surgery",
    "Dermatology",
    "Neurology",
    "Emergency Medicine",
    "Gastroenterology",
    "Urology",
    "Pulmonology",
    "Endocrinology",
    "Ophthalmology",
]

_STATE_OPTIONS = [
    "", "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE",
    "DC", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS",
    "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO",
    "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND",
    "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX",
    "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]

_INSURANCE_OPTIONS = [
    "",
    "Medicare",
    "Medicaid",
    "Blue Cross Blue Shield",
    "Aetna",
    "Cigna",
    "Humana",
    "UnitedHealth",
    "Tricare",
    "Kaiser Permanente",
]


def _init_state() -> None:
    defaults: Dict[str, Any] = {
        "session_id": str(uuid.uuid4()),
        "messages": [],
        "query_history": [],
        "total_queries": 0,
        "cache_hits": 0,
        "_prefill": "",
    }
    for key, default_val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_val


def _build_sidebar() -> Dict[str, Any]:
    with st.sidebar:
        st.markdown(
            "## :hospital: Provider Search\n"
            '<span class="synth-badge">'
            "SYNTHETIC DATA ONLY"
            "</span>",
            unsafe_allow_html=True,
        )
        st.caption(
            "All provider data is synthetically generated. "
            "No real patient or provider information is used."
        )
        st.divider()

        st.subheader("Filters")
        chosen_specialty = st.selectbox(
            "Specialty", _SPECIALTY_OPTIONS
        )
        chosen_state = st.selectbox(
            "State", _STATE_OPTIONS
        )
        chosen_insurance = st.selectbox(
            "Insurance", _INSURANCE_OPTIONS
        )
        accepting_only = st.checkbox(
            "Accepting new patients only", value=False
        )
        use_hyde = st.checkbox(
            "Enable HyDE (better recall, +500ms)",
            value=False,
            help=(
                "Generates a hypothetical provider bio to "
                "improve retrieval quality for vague queries."
            ),
        )
        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.messages = []
                st.session_state.query_history = []
                st.session_state.total_queries = 0
                st.session_state.cache_hits = 0
                st.rerun()
        with c2:
            export_payload = json.dumps(
                st.session_state.messages,
                indent=2,
                default=str,
            )
            st.download_button(
                "Export",
                export_payload,
                file_name="chat_history.json",
                mime="application/json",
                use_container_width=True,
            )

        total = st.session_state.total_queries
        if total > 0:
            hit_rate = (
                st.session_state.cache_hits / total * 100
            )
            st.metric("Cache Hit Rate", f"{hit_rate:.0f}%")

        history = st.session_state.query_history
        if history:
            st.divider()
            st.caption("Recent queries")
            for past_q in reversed(history[-8:]):
                label = (
                    past_q[:44] + "..."
                    if len(past_q) > 44
                    else past_q
                )
                if st.button(
                    label,
                    key=f"hist_{hash(past_q)}",
                    use_container_width=True,
                ):
                    st.session_state["_prefill"] = past_q
                    st.rerun()

        active_filters: Dict[str, Any] = {}
        if chosen_specialty:
            active_filters["specialty"] = chosen_specialty
        if chosen_state:
            active_filters["state"] = chosen_state
        if chosen_insurance:
            active_filters["insurance"] = chosen_insurance
        if accepting_only:
            active_filters["accepting_new_patients"] = True

        return {
            "filters": active_filters if active_filters else None,
            "hyde_enabled": use_hyde,
        }


def _render_assistant_turn(msg: Dict[str, Any]) -> None:
    answer = msg.get("content", "")
    data = msg.get("response_data")
    cache_hit = msg.get("cache_hit", False)
    latency_ms = msg.get("latency_ms", 0)
    model_used = msg.get("model_used", "")

    render_response_header(cache_hit, latency_ms, model_used)

    if "⚠️" in answer or (
        data and data.get("_hallucination_warning")
    ):
        render_hallucination_warning()

    st.markdown(answer)

    if not data:
        return

    sources: List[Dict[str, Any]] = data.get("sources", [])
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            render_sources_table(sources)

    map_data: List[Dict[str, Any]] = data.get(
        "providers_map", []
    )
    if map_data:
        with st.expander("Provider Map"):
            render_geo_map(map_data)

    suggestions: List[str] = data.get("suggestions", [])
    render_suggestion_buttons(
        suggestions, key_prefix=str(id(msg))
    )


def _post_query(
    query: str,
    session_id: str,
    filters: Optional[Dict[str, Any]],
    hyde_enabled: bool,
) -> Optional[Dict[str, Any]]:
    """Call /query synchronously from a Streamlit context."""

    async def _async_post() -> Dict[str, Any]:
        payload = {
            "query": query,
            "session_id": session_id,
            "filters": filters,
            "hyde_enabled": hyde_enabled,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{API_BASE}/query", json=payload
            )
            resp.raise_for_status()
            return resp.json()

    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=1
        ) as pool:
            future = pool.submit(asyncio.run, _async_post())
            return future.result(timeout=35)
    except httpx.TimeoutException:
        st.error("Request timed out. Please try again.")
    except httpx.ConnectError:
        st.error(
            f"Cannot connect to API at {API_BASE}. "
            "Is the backend running?"
        )
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        if code == 429:
            st.warning("Rate limit reached. Wait then retry.")
        elif code == 503:
            st.error("Service unavailable. Try again later.")
        else:
            st.error(f"API error {code}.")
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")

    return None


def main() -> None:
    _init_state()
    sidebar_cfg = _build_sidebar()

    st.title(":hospital: Healthcare Provider Search")
    st.caption(
        "Natural language search over 10,000 synthetic providers. "
        "All data is fictional — for demonstration only."
    )

    for turn in st.session_state.messages:
        if turn["role"] == "user":
            with st.chat_message("user"):
                st.markdown(turn["content"])
        else:
            with st.chat_message("assistant"):
                _render_assistant_turn(turn)

    current_prefill = st.session_state.pop("_prefill", "")

    with st.form("chat_form", clear_on_submit=True):
        col_txt, col_btn = st.columns([5, 1])
        with col_txt:
            raw_input = st.text_area(
                "Your query",
                value=current_prefill,
                placeholder=(
                    "e.g. Top-rated cardiologists in California "
                    "accepting Medicare and new patients"
                ),
                height=80,
                label_visibility="collapsed",
            )
        with col_btn:
            was_submitted = st.form_submit_button(
                "Ask",
                use_container_width=True,
                type="primary",
            )

    if was_submitted and raw_input and raw_input.strip():
        clean_query = raw_input.strip()
        st.session_state.messages.append(
            {"role": "user", "content": clean_query}
        )
        if clean_query not in st.session_state.query_history:
            st.session_state.query_history.append(clean_query)

        with st.spinner("Searching providers..."):
            api_result = _post_query(
                query=clean_query,
                session_id=st.session_state.session_id,
                filters=sidebar_cfg.get("filters"),
                hyde_enabled=sidebar_cfg.get(
                    "hyde_enabled", False
                ),
            )

        if api_result:
            returned_sid = api_result.get("session_id")
            if returned_sid:
                st.session_state.session_id = str(returned_sid)

            st.session_state.total_queries += 1
            if api_result.get("cache_hit"):
                st.session_state.cache_hits += 1

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": api_result.get("answer", ""),
                    "response_data": api_result,
                    "cache_hit": api_result.get(
                        "cache_hit", False
                    ),
                    "latency_ms": api_result.get(
                        "latency_ms", 0
                    ),
                    "model_used": api_result.get(
                        "model_used", ""
                    ),
                }
            )
        else:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "Sorry, I could not process your request. "
                        "Please try again."
                    ),
                }
            )

        st.rerun()


if __name__ == "__main__":
    main()
