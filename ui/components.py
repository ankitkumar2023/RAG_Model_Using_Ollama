from __future__ import annotations

from ui.theme import COLORS, GRADIENT_PRIMARY, ROUTE_ICONS, ROUTE_LABELS

"""
Reusable design-system components.

Each function returns an HTML string built from the design tokens in
ui.theme -- nothing here hardcodes a color/radius/spacing value that
isn't already in the token set, so every component stays visually
consistent automatically. Callers render with `st.html(html)` --
NOT `st.markdown(html, unsafe_allow_html=True)`. Markdown's CommonMark
parser treats any line indented 4+ spaces as a literal (escaped) code
block regardless of unsafe_allow_html, which is exactly what caused
raw "<div style=..." tags to show up on-screen instead of rendering.
`st.html()` skips Markdown parsing entirely, so it doesn't matter.

`_flatten()` below strips leading whitespace from every line as a
second, defensive layer -- these strings then render correctly even if
some future call site accidentally routes one through st.markdown.
"""


def _flatten(html: str) -> str:
    """
    Collapse an HTML fragment to one tag/line of content per line, no
    leading whitespace. HTML itself doesn't care about whitespace
    between tags, so this is always safe.
    """

    return "\n".join(line.strip() for line in html.strip().splitlines() if line.strip())


def ai_card(
    title: str,
    content_html: str,
    icon: str = "",
    accent: bool = False,
) -> str:
    border = f"1px solid {COLORS['primary']}" if accent else f"1px solid {COLORS['border']}"

    return _flatten(f"""
    <div style="
        background: {COLORS["glass"]};
        backdrop-filter: blur(12px);
        border: {border};
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 12px;
        animation: fadeInUp 0.35s ease both;
    ">
        <div style="
            font-size: 13px;
            font-weight: 600;
            color: {COLORS["text_secondary"]};
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
        ">{icon + " " if icon else ""}{title}</div>
        <div style="color: {COLORS["text_primary"]};">{content_html}</div>
    </div>
    """)


def metric_card(label: str, value: str, icon: str = "") -> str:
    return _flatten(f"""
    <div style="
        background: {COLORS["surface_elevated"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        transition: all 0.2s ease;
    ">
        <div style="font-size: 22px; margin-bottom: 4px;">{icon}</div>
        <div style="
            font-size: 20px;
            font-weight: 700;
            background: {GRADIENT_PRIMARY};
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        ">{value}</div>
        <div style="
            font-size: 12px;
            color: {COLORS["text_secondary"]};
            margin-top: 2px;
        ">{label}</div>
    </div>
    """)


def status_badge(label: str, status: str = "online") -> str:
    color_map = {
        "online": COLORS["success"],
        "offline": COLORS["error"],
        "warning": COLORS["warning"],
    }
    color = color_map.get(status, COLORS["text_muted"])
    pulse = "animation: pulse 2s ease-in-out infinite;" if status == "online" else ""

    return _flatten(f"""
    <span style="
        display: inline-flex; align-items: center; gap: 6px;
        background: {COLORS["surface_elevated"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 999px; padding: 4px 12px;
        font-size: 12px; color: {COLORS["text_secondary"]};
    ">
        <span style="width: 7px; height: 7px; border-radius: 999px; background: {color}; {pulse}"></span>
        {label}
    </span>
    """)


def tool_badge(route: str) -> str:
    icon = ROUTE_ICONS.get(route, "🧠")
    label = ROUTE_LABELS.get(route, route.replace("_", " ").title())

    return _flatten(f"""
    <span style="
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(124, 58, 237, 0.12);
        border: 1px solid rgba(124, 58, 237, 0.35);
        border-radius: 999px; padding: 4px 12px;
        font-size: 12px; font-weight: 500; color: {COLORS["text_primary"]};
    ">{icon} {label}</span>
    """)


def confidence_meter(score: float, label: str = "Confidence") -> str:
    score = max(0.0, min(1.0, score))
    pct = round(score * 100)

    if score >= 0.7:
        color = COLORS["success"]
    elif score >= 0.4:
        color = COLORS["warning"]
    else:
        color = COLORS["error"]

    return _flatten(f"""
    <div style="margin: 6px 0;">
        <div style="
            display: flex; justify-content: space-between;
            font-size: 12px; color: {COLORS["text_secondary"]};
            margin-bottom: 4px;
        ">
            <span>{label}</span><span>{pct}%</span>
        </div>
        <div style="
            width: 100%; height: 6px; border-radius: 999px;
            background: {COLORS["surface_elevated"]}; overflow: hidden;
        ">
            <div style="
                width: {pct}%; height: 100%; border-radius: 999px;
                background: {color}; transition: width 0.6s ease;
            "></div>
        </div>
    </div>
    """)


def citation_card(
    filename: str,
    page: int | None,
    score: float,
    snippet: str,
) -> str:
    page_label = f" &middot; Page {page}" if page else ""
    snippet_trimmed = (snippet[:220] + "...") if len(snippet) > 220 else snippet

    return _flatten(f"""
    <div style="
        background: {COLORS["surface_elevated"]};
        border: 1px solid {COLORS["border"]};
        border-left: 3px solid {COLORS["accent"]};
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 8px;
        transition: transform 0.15s ease, border-color 0.15s ease;
    " onmouseover="this.style.transform='translateX(2px)'" onmouseout="this.style.transform='translateX(0)'">
        <div style="
            display: flex; justify-content: space-between; align-items: center;
            font-size: 13px; font-weight: 600; color: {COLORS["text_primary"]};
            margin-bottom: 6px;
        ">
            <span>📄 {filename}{page_label}</span>
            <span style="
                font-size: 11px; color: {COLORS["text_secondary"]};
                background: rgba(0,212,255,0.1); padding: 2px 8px; border-radius: 999px;
            ">{score:.0%} match</span>
        </div>
        <div style="font-size: 13px; color: {COLORS["text_secondary"]}; line-height: 1.5;">
            {snippet_trimmed}
        </div>
    </div>
    """)


def hero_banner(
    title: str,
    subtitle: str,
    status_items: list[tuple[str, str, str]],
) -> str:
    """
    status_items: list of (icon, label, value) triples.
    """

    items_html = "".join(
        f"""<div style="
            display:flex; align-items:center; gap:6px;
            background: {COLORS["surface_elevated"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 999px; padding: 6px 14px; font-size: 12px;
            color: {COLORS["text_secondary"]};
            transition: border-color 0.2s ease;
        ">{icon} {label} <b style="color:{COLORS["text_primary"]};">{value}</b></div>"""
        for icon, label, value in status_items
    )

    return _flatten(f"""
    <div style="
        background: radial-gradient(circle at 20% 20%, rgba(124,58,237,0.18), transparent 55%),
                    radial-gradient(circle at 80% 30%, rgba(0,212,255,0.14), transparent 50%),
                    {COLORS["surface"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 20px;
        animation: fadeIn 0.5s ease both;
    ">
        <div style="
            font-size: 30px; font-weight: 800; letter-spacing: -0.02em;
            background: {GRADIENT_PRIMARY};
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-size: 200% auto;
            animation: shimmer 6s linear infinite;
        ">{title}</div>
        <div style="color: {COLORS["text_secondary"]}; margin-top: 4px; font-size: 14px;">
            {subtitle}
        </div>
        <div style="display:flex; flex-wrap:wrap; gap:8px; margin-top:16px;">
            {items_html}
        </div>
    </div>
    """)


def loading_pipeline(stages: list[str], current_index: int) -> str:
    """
    Horizontal animated stage tracker. `current_index` is the index of
    the stage currently in progress; earlier stages render as complete.
    """

    items = []

    for idx, stage in enumerate(stages):
        if idx < current_index:
            icon, color, weight = "✓", COLORS["success"], "500"
        elif idx == current_index:
            icon, color, weight = "●", COLORS["accent"], "700"
        else:
            icon, color, weight = "○", COLORS["text_muted"], "400"

        pulse = "animation: pulse 1.2s ease-in-out infinite;" if idx == current_index else ""

        items.append(
            f"""<div style="display:flex; align-items:center; gap:8px;">"""
            f"""<span style="color:{color}; {pulse} font-size:14px;">{icon}</span>"""
            f"""<span style="color:{color}; font-weight:{weight}; font-size:13px;">{stage}</span>"""
            f"""</div>"""
        )

    return _flatten(f"""
    <div style="
        background: {COLORS["surface_elevated"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 12px;
        padding: 14px 18px;
        display: flex; flex-direction: column; gap: 10px;
        margin-bottom: 8px;
    ">
        {"".join(items)}
    </div>
    """)


def upload_card(
    filename: str,
    size_label: str,
    status_label: str,
    icon: str = "📄",
) -> str:
    return _flatten(f"""
    <div style="
        display:flex; align-items:center; justify-content:space-between;
        background: {COLORS["surface_elevated"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 6px;
        animation: fadeInUp 0.3s ease both;
    ">
        <div style="display:flex; align-items:center; gap:10px;">
            <span style="font-size:18px;">{icon}</span>
            <div>
                <div style="font-size:13px; color:{COLORS["text_primary"]}; font-weight:500;">{filename}</div>
                <div style="font-size:11px; color:{COLORS["text_muted"]};">{size_label}</div>
            </div>
        </div>
        <span style="font-size:11px; color:{COLORS["success"]};">{status_label}</span>
    </div>
    """)


def status_bar_html(
    gemini_online: bool,
    route: str | None,
    latency_ms: float | None,
    confidence: float | None,
) -> str:
    gemini_dot = COLORS["success"] if gemini_online else COLORS["error"]

    route_html = ""
    if route:
        icon = ROUTE_ICONS.get(route, "🧠")
        label = ROUTE_LABELS.get(route, route.replace("_", " ").title())
        route_html = f"<span>{icon} {label}</span>"

    latency_html = f"<span>⚡ {latency_ms:.0f}ms</span>" if latency_ms else ""
    confidence_html = f"<span>🎯 {confidence:.0%} confidence</span>" if confidence is not None else ""

    return _flatten(f"""
    <div style="
        position: fixed; bottom: 0; left: 0; right: 0; z-index: 999;
        background: {COLORS["glass"]};
        backdrop-filter: blur(16px);
        border-top: 1px solid {COLORS["border"]};
        padding: 8px 24px;
        display: flex; gap: 20px; align-items: center;
        font-size: 12px; color: {COLORS["text_secondary"]};
    ">
        <span style="display:flex; align-items:center; gap:6px;">
            <span style="width:7px; height:7px; border-radius:999px; background:{gemini_dot};"></span>
            Gemini
        </span>
        {route_html}
        {latency_html}
        {confidence_html}
    </div>
    """)


_TIMELINE_STAGE_ICONS = {
    "Query Routed": "🧠",
    "Retrieval": "📄",
    "Web Search": "🌐",
    "Corrective Fallback": "⚡",
    "Tools": "🧮",
    "Generation": "✨",
}


def router_timeline_card(
    route: str,
    reason: str,
    confidence: float,
    tools_used: list[str],
    latency_ms: dict[str, float],
    corrective_fallback: bool,
    reranked: bool,
    retrieved_documents: int,
    web_results_used: int,
) -> str:
    """
    Premium timeline-style Router Inspector panel: glowing connected
    nodes, one per pipeline stage that actually ran for this query.

    Built on a native HTML `<details>` element (not `st.expander`) so it
    stays fully custom-styled and collapsible without pulling in
    Streamlit's own widget chrome.
    """

    icon = ROUTE_ICONS.get(route, "🧠")
    label = ROUTE_LABELS.get(route, route.replace("_", " ").title())

    nodes = [("Query Routed", f"<b>{label}</b> &mdash; {reason}")]

    if route in ("document_rag", "hybrid"):
        nodes.append(
            (
                "Retrieval",
                (
                    f"{retrieved_documents} chunk(s) retrieved &middot; "
                    f"reranked: {'yes' if reranked else 'no'}"
                ),
            )
        )

    if route in ("web_search", "hybrid") or corrective_fallback:
        nodes.append(("Web Search", f"{web_results_used} result(s) retrieved"))

    if corrective_fallback:
        nodes.append(
            (
                "Corrective Fallback",
                ("Retrieval confidence was low -- web search was automatically added."),
            )
        )

    if tools_used:
        nodes.append(("Tools", ", ".join(f"✓ {t}" for t in tools_used)))

    nodes.append(
        (
            "Generation",
            (f"Final answer generated in {latency_ms.get('Generation', 0):.0f}ms"),
        )
    )

    node_blocks = []

    for i, (title, detail) in enumerate(nodes):
        node_icon = _TIMELINE_STAGE_ICONS.get(title, "•")
        delay = i * 0.08

        node_blocks.append(
            f"""<div class="timeline-node" style="animation-delay:{delay}s;">"""
            f"""<div style="font-size:12px; font-weight:700; color:{COLORS["text_primary"]};">"""
            f"""{node_icon} {title}</div>"""
            f"""<div style="font-size:12px; color:{COLORS["text_secondary"]}; margin-top:2px;">{detail}</div>"""
            f"""</div>"""
        )

    nodes_html = "".join(node_blocks)

    latency_chips = "".join(
        f"""<span style="
            background:{COLORS["surface_elevated"]}; border-radius:999px;
            padding:3px 10px; font-size:11px; color:{COLORS["text_secondary"]}; margin-right:6px;
        ">{name}: <b style="color:{COLORS["text_primary"]}">{value:.0f}ms</b></span>"""
        for name, value in latency_ms.items()
    )

    return _flatten(f"""
    <details class="router-timeline">
        <summary>🧠 Router Decision &mdash; {icon} {label} &middot; {confidence:.0%} confidence</summary>
        <div class="timeline-track">
            {nodes_html}
        </div>
        <div style="margin-top:10px;">{confidence_meter(confidence, label="Route confidence")}</div>
        <div style="margin-top:10px;">{latency_chips}</div>
    </details>
    """)


def diagnostics_dashboard(
    latency_ms: float,
    approx_tokens: int,
    streaming_speed: float,
    api_calls: int,
    retrieved_documents: int,
    confidence: float,
    model: str,
    route: str = "",
    sources: list[dict] | None = None,
) -> str:
    """
    Live diagnostics dashboard (Developer Mode). Grid of metric cards
    built from the same tokens as everything else.
    """

    sources = sources or []
    source_names = (
        ", ".join(dict.fromkeys(s.get("filename", "") for s in sources if s.get("filename"))) or "--"
    )
    route_label = ROUTE_LABELS.get(route, route.replace("_", " ").title() if route else "--")

    cells = [
        ("⚡", f"{latency_ms:.0f}ms", "Latency"),
        ("🔤", f"~{approx_tokens}", "Tokens"),
        ("🚀", f"{streaming_speed:.0f} tok/s", "Streaming Speed"),
        ("📡", str(api_calls), "API Calls"),
        ("📄", str(retrieved_documents), "Chunks"),
        ("🎯", f"{confidence:.0%}", "Confidence"),
        ("🧭", route_label, "Route"),
        ("📚", source_names, "Sources"),
    ]

    cells_html = "".join(
        f"""<div style="
            background:{COLORS["surface_elevated"]}; border:1px solid {COLORS["border"]};
            border-radius:10px; padding:10px; text-align:center;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        " onmouseover="this.style.transform='translateY(-2px)'" onmouseout="this.style.transform='translateY(0)'">"""
        f"""<div style="font-size:16px;">{icon}</div>"""
        f"""<div style="font-size:13px; font-weight:700; color:{COLORS["text_primary"]}; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{value}</div>"""
        f"""<div style="font-size:10px; color:{COLORS["text_secondary"]};">{label}</div>"""
        f"""</div>"""
        for icon, value, label in cells
    )

    return _flatten(f"""
    <div style="
        background:{COLORS["glass"]}; backdrop-filter: blur(12px);
        border:1px solid {COLORS["border"]}; border-radius:16px; padding:16px; margin-top:10px;
    ">
        <div style="font-size:12px; font-weight:700; color:{COLORS["text_secondary"]}; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:8px;">
            🛠️ Developer Diagnostics &middot; {model}
        </div>
        <div class="diag-grid">{cells_html}</div>
    </div>
    """)
