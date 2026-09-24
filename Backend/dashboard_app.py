"""
Streamlit dashboard for ChainGuard.

Run:
  streamlit run dashboard_app.py
"""
from __future__ import annotations

import html
from typing import Any

import httpx
import pandas as pd
import streamlit as st

API_DEFAULT = "http://127.0.0.1:8000"


def _risk_color(score: int) -> str:
    if score >= 75:
        return "#ef4444"
    if score >= 45:
        return "#f59e0b"
    return "#10b981"


def _build_dot(subgraph: dict[str, Any]) -> str:
    nodes = subgraph.get("nodes", [])
    edges = subgraph.get("edges", [])

    lines = ["digraph G {", "rankdir=LR;", "bgcolor=\"transparent\";"]
    for n in nodes:
        node_id = html.escape(str(n.get("id", "")))
        label = node_id[:8] + "..." + node_id[-6:] if len(node_id) > 18 else node_id
        if n.get("is_target"):
            color = "#00d4ff"
            shape = "doublecircle"
        elif n.get("node_type") == "contract":
            color = "#ef4444"
            shape = "box"
        else:
            color = "#94a3b8"
            shape = "circle"
        lines.append(
            f'"{node_id}" [label="{label}", color="{color}", shape="{shape}", fontcolor="{color}"];'
        )

    for e in edges:
        src = html.escape(str(e.get("source", "")))
        dst = html.escape(str(e.get("target", "")))
        if src and dst:
            lines.append(f'"{src}" -> "{dst}" [color="#334155"];')

    lines.append("}")
    return "\n".join(lines)


def _call_analyze(api_base: str, address: str, force_refresh: bool) -> dict[str, Any]:
    payload = {"address": address, "force_refresh": force_refresh}
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(f"{api_base}/api/v1/analyze", json=payload)
        resp.raise_for_status()
        return resp.json()


def main() -> None:
    st.set_page_config(page_title="ChainGuard Dashboard", layout="wide")
    st.title("ChainGuard Risk Dashboard")
    st.caption("Analyze an Ethereum address and inspect risk score, graph, suspicious neighbors, and explanations.")

    with st.sidebar:
        st.header("Settings")
        api_base = st.text_input("Backend URL", value=API_DEFAULT)
        force_refresh = st.checkbox("Force refresh", value=False)

        if st.button("Check API health"):
            try:
                with httpx.Client(timeout=5.0) as client:
                    health = client.get(f"{api_base}/health")
                    health.raise_for_status()
                st.success("API is online")
                st.json(health.json())
            except Exception as exc:  # noqa: BLE001
                st.error(f"API health check failed: {exc}")

    address = st.text_input("Ethereum Address or ENS", placeholder="0x...")

    if st.button("Analyze", type="primary"):
        if not address.strip():
            st.warning("Please provide an address.")
            st.stop()

        try:
            with st.spinner("Running HT-GNN analysis..."):
                data = _call_analyze(api_base, address.strip(), force_refresh)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Analysis failed: {exc}")
            st.stop()

        score = int(data.get("phishing_score", 0))
        verdict = str(data.get("verdict", "unknown")).upper()
        risk_tier = str(data.get("risk_tier", "UNKNOWN"))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Risk Score", f"{score}%")
        c2.metric("Verdict", verdict)
        c3.metric("Risk Tier", risk_tier)
        c4.metric("Confidence", f"{round(float(data.get('confidence', 0.0)) * 100, 1)}%")

        st.markdown(
            f"<div style='height:10px;border-radius:8px;background:{_risk_color(score)};margin:8px 0 16px 0'></div>",
            unsafe_allow_html=True,
        )

        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.subheader("Model Explanation")
            st.markdown(data.get("explanation", "No explanation returned."), unsafe_allow_html=True)

            st.subheader("Evidence")
            evidence = data.get("evidence", [])
            if evidence:
                ev_df = pd.DataFrame(evidence)
                st.dataframe(ev_df, use_container_width=True, hide_index=True)
            else:
                st.info("No evidence items returned.")

            st.subheader("Feature Importance")
            fi = data.get("feature_importance", [])
            if fi:
                fi_df = pd.DataFrame(fi)
                st.dataframe(fi_df, use_container_width=True, hide_index=True)
            else:
                st.info("No feature importance returned.")

        with col_b:
            st.subheader("Suspicious Neighbors")
            neighbors = data.get("suspicious_neighbors", [])
            if neighbors:
                n_df = pd.DataFrame(neighbors)
                st.dataframe(n_df, use_container_width=True, hide_index=True)
                st.metric("Detected", len(neighbors))
            else:
                st.success("No suspicious neighbors detected by heuristic thresholds.")

            st.subheader("Quick Stats")
            st.json(
                {
                    "num_hop1_neighbors": data.get("num_hop1_neighbors"),
                    "num_contracts": data.get("num_contracts"),
                    "num_inflow_txs": data.get("num_inflow_txs"),
                    "num_approvals": data.get("num_approvals"),
                    "model_used": data.get("model_used"),
                    "fetch_time_s": data.get("fetch_time_s"),
                    "inference_time_ms": data.get("inference_time_ms"),
                    "cached": data.get("cached"),
                }
            )

        st.subheader("Transaction Subgraph")
        subgraph = data.get("subgraph", {})
        dot = _build_dot(subgraph)
        st.graphviz_chart(dot, use_container_width=True)

        nodes = pd.DataFrame(subgraph.get("nodes", []))
        edges = pd.DataFrame(subgraph.get("edges", []))
        ncol, ecol = st.columns(2)
        with ncol:
            st.caption("Nodes")
            st.dataframe(nodes, use_container_width=True, hide_index=True)
        with ecol:
            st.caption("Edges")
            st.dataframe(edges, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
