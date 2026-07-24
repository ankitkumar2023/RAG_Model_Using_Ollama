from __future__ import annotations

import streamlit as st

# =============================================================
# DESIGN TOKENS
# =============================================================
# Single source of truth for the design system. Every component in
# ui/components.py pulls colors/spacing/radius from here rather than
# hardcoding values, so the whole app stays visually consistent.

COLORS = {
    "primary": "#7C3AED",  # purple accent
    "primary_hover": "#8B5CF6",
    "accent": "#00D4FF",  # blue accent
    "emerald": "#10B981",  # emerald accent
    "background": "#0B1020",
    "surface": "#161B2E",
    "surface_elevated": "#1E2438",
    "border": "rgba(255, 255, 255, 0.08)",
    "border_strong": "rgba(255, 255, 255, 0.16)",
    "text_primary": "#F5F5F7",
    "text_secondary": "#9CA3AF",
    "text_muted": "#6B7280",
    "success": "#10B981",
    "warning": "#F59E0B",
    "error": "#EF4444",
    "glass": "rgba(22, 27, 46, 0.65)",
}

GRADIENT_PRIMARY = f"linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['accent']} 100%)"

GRADIENT_AURORA = (
    f"linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['accent']} 50%, {COLORS['emerald']} 100%)"
)

SPACING = {
    "xs": "4px",
    "sm": "8px",
    "md": "16px",
    "lg": "24px",
    "xl": "32px",
    "2xl": "48px",
}

RADIUS = {
    "sm": "8px",
    "md": "12px",
    "lg": "16px",
    "xl": "24px",
    "full": "999px",
}

SHADOWS = {
    "sm": "0 2px 8px rgba(0, 0, 0, 0.24)",
    "md": "0 8px 24px rgba(0, 0, 0, 0.32)",
    "lg": "0 16px 48px rgba(0, 0, 0, 0.4)",
    "glow": "0 0 24px rgba(124, 58, 237, 0.35)",
}

ROUTE_ICONS = {
    "document_rag": "📄",
    "web_search": "🌐",
    "weather": "☀️",
    "finance": "📈",
    "general": "🧠",
    "hybrid": "🧬",
    "tool_calling": "🧮",
}

ROUTE_LABELS = {
    "document_rag": "Documents",
    "web_search": "Web",
    "weather": "Weather",
    "finance": "Finance",
    "general": "Gemini Knowledge",
    "hybrid": "Documents + Web",
    "tool_calling": "Tool",
}


def render_aurora_background() -> None:
    """
    Fixed-position animated gradient-mesh background blobs. Call once,
    anywhere in the script -- `position: fixed` means DOM placement
    doesn't matter.
    """

    st.markdown(
        """
<div class="aurora-layer">
    <div class="aurora-blob a1"></div>
    <div class="aurora-blob a2"></div>
    <div class="aurora-blob a3"></div>
</div>
""",
        unsafe_allow_html=True,
    )


def inject_theme() -> None:
    """
    Inject the design system as a single global stylesheet.

    Called once per script run. Overrides Streamlit's default chrome
    (hides the hamburger menu/footer, restyles chat/buttons/inputs/
    sidebar/expanders/file uploader) rather than leaving default widget
    styling in place.
    """

    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {{
    --color-primary: {COLORS["primary"]};
    --color-primary-hover: {COLORS["primary_hover"]};
    --color-accent: {COLORS["accent"]};
    --color-background: {COLORS["background"]};
    --color-surface: {COLORS["surface"]};
    --color-surface-elevated: {COLORS["surface_elevated"]};
    --color-border: {COLORS["border"]};
    --color-border-strong: {COLORS["border_strong"]};
    --color-text-primary: {COLORS["text_primary"]};
    --color-text-secondary: {COLORS["text_secondary"]};
    --color-text-muted: {COLORS["text_muted"]};
    --color-success: {COLORS["success"]};
    --color-warning: {COLORS["warning"]};
    --color-error: {COLORS["error"]};
    --color-glass: {COLORS["glass"]};
    --color-emerald: {COLORS["emerald"]};
    --gradient-primary: {GRADIENT_PRIMARY};
    --gradient-aurora: {GRADIENT_AURORA};
    --radius-sm: {RADIUS["sm"]};
    --radius-md: {RADIUS["md"]};
    --radius-lg: {RADIUS["lg"]};
    --radius-xl: {RADIUS["xl"]};
    --radius-full: {RADIUS["full"]};
    --shadow-sm: {SHADOWS["sm"]};
    --shadow-md: {SHADOWS["md"]};
    --shadow-lg: {SHADOWS["lg"]};
    --shadow-glow: {SHADOWS["glow"]};
    --space-xs: {SPACING["xs"]};
    --space-sm: {SPACING["sm"]};
    --space-md: {SPACING["md"]};
    --space-lg: {SPACING["lg"]};
    --space-xl: {SPACING["xl"]};
    --space-2xl: {SPACING["2xl"]};
}}

/* ---------- Global ---------- */

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}}

.stApp {{
    background:
        radial-gradient(circle at 15% 0%, rgba(124, 58, 237, 0.12) 0%, transparent 45%),
        radial-gradient(circle at 85% 20%, rgba(0, 212, 255, 0.08) 0%, transparent 45%),
        var(--color-background);
    color: var(--color-text-primary);
}}

code, pre, .stCode {{
    font-family: 'JetBrains Mono', monospace !important;
}}

/* Hide default Streamlit chrome */
#MainMenu, footer, header[data-testid="stHeader"] {{
    visibility: hidden;
    height: 0;
}}

div[data-testid="stDecoration"] {{
    display: none;
}}

.block-container {{
    padding-top: var(--space-lg) !important;
    padding-bottom: 6rem !important;
    max-width: 1200px;
}}

::-webkit-scrollbar {{
    width: 8px;
    height: 8px;
}}

::-webkit-scrollbar-track {{
    background: transparent;
}}

::-webkit-scrollbar-thumb {{
    background: var(--color-border-strong);
    border-radius: var(--radius-full);
}}

::-webkit-scrollbar-thumb:hover {{
    background: var(--color-primary);
}}

/* ---------- Sidebar ---------- */

section[data-testid="stSidebar"] {{
    background: var(--color-surface);
    border-right: 1px solid var(--color-border);
}}

section[data-testid="stSidebar"] .block-container {{
    padding-top: var(--space-lg);
}}

/* ---------- Buttons ---------- */

.stButton > button, .stDownloadButton > button {{
    background: var(--color-surface-elevated);
    color: var(--color-text-primary);
    border: 1px solid var(--color-border-strong);
    border-radius: var(--radius-md);
    font-weight: 500;
    transition: all 0.2s ease;
}}

.stButton > button:hover, .stDownloadButton > button:hover {{
    border-color: var(--color-primary);
    box-shadow: var(--shadow-glow);
    transform: translateY(-1px);
}}

.stButton > button[kind="primary"] {{
    background: var(--gradient-primary);
    border: none;
}}

/* ---------- Inputs / chat input ---------- */

.stTextInput input, .stTextArea textarea, .stNumberInput input,
div[data-baseweb="select"] > div {{
    background: var(--color-surface-elevated) !important;
    color: var(--color-text-primary) !important;
    border: 1px solid var(--color-border-strong) !important;
    border-radius: var(--radius-md) !important;
}}

div[data-testid="stChatInput"] {{
    background: var(--color-glass);
    backdrop-filter: blur(16px);
    border: 1px solid var(--color-border-strong);
    border-radius: var(--radius-lg);
}}

div[data-testid="stChatInput"] textarea {{
    color: var(--color-text-primary) !important;
}}

/* ---------- Chat messages ---------- */

div[data-testid="stChatMessage"] {{
    background: var(--color-glass);
    backdrop-filter: blur(12px);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-lg);
    padding: var(--space-md);
    margin-bottom: var(--space-md);
    animation: fadeInUp 0.35s ease both;
}}

/* ---------- Expanders (used for citations / router inspector) ---------- */

.streamlit-expanderHeader, div[data-testid="stExpander"] summary {{
    background: var(--color-surface-elevated) !important;
    border-radius: var(--radius-md) !important;
    color: var(--color-text-secondary) !important;
}}

div[data-testid="stExpander"] {{
    border: 1px solid var(--color-border) !important;
    border-radius: var(--radius-md) !important;
    background: transparent !important;
}}

/* ---------- File uploader ---------- */

div[data-testid="stFileUploaderDropzone"] {{
    background: var(--color-surface-elevated);
    border: 1.5px dashed var(--color-border-strong);
    border-radius: var(--radius-lg);
    transition: all 0.2s ease;
}}

div[data-testid="stFileUploaderDropzone"]:hover {{
    border-color: var(--color-accent);
    box-shadow: 0 0 16px rgba(0, 212, 255, 0.2);
}}

/* ---------- Status / alert boxes ---------- */

div[data-testid="stAlert"] {{
    background: var(--color-surface-elevated);
    border-radius: var(--radius-md);
    border: 1px solid var(--color-border);
}}

/* ---------- Animations ---------- */

@keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(8px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

@keyframes fadeIn {{
    from {{ opacity: 0; }}
    to {{ opacity: 1; }}
}}

@keyframes shimmer {{
    0% {{ background-position: -200% 0; }}
    100% {{ background-position: 200% 0; }}
}}

@keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.4; }}
}}

@keyframes blink {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0; }}
}}

@keyframes glow {{
    0%, 100% {{ box-shadow: 0 0 8px rgba(124, 58, 237, 0.4); }}
    50% {{ box-shadow: 0 0 20px rgba(124, 58, 237, 0.7); }}
}}

.streaming-cursor {{
    display: inline-block;
    width: 8px;
    height: 1em;
    background: var(--color-accent);
    margin-left: 2px;
    vertical-align: text-bottom;
    animation: blink 0.9s step-start infinite;
}}

/* ---------- Aurora / mesh background ---------- */

.aurora-layer {{
    position: fixed;
    inset: 0;
    z-index: -1;
    overflow: hidden;
    pointer-events: none;
}}

.aurora-blob {{
    position: absolute;
    width: 42vw;
    height: 42vw;
    border-radius: 50%;
    filter: blur(90px);
    opacity: 0.28;
    animation: auroraDrift 22s ease-in-out infinite alternate;
}}

.aurora-blob.a1 {{
    top: -10%; left: -8%;
    background: var(--color-primary);
}}

.aurora-blob.a2 {{
    top: 20%; right: -12%;
    background: var(--color-accent);
    animation-duration: 28s;
    animation-delay: -6s;
}}

.aurora-blob.a3 {{
    bottom: -15%; left: 30%;
    background: var(--color-emerald);
    animation-duration: 34s;
    animation-delay: -12s;
}}

@keyframes auroraDrift {{
    0%   {{ transform: translate(0, 0) scale(1); }}
    50%  {{ transform: translate(4%, 6%) scale(1.08); }}
    100% {{ transform: translate(-3%, -4%) scale(0.96); }}
}}

/* ---------- Animated gradient border utility ---------- */

.gradient-border {{
    position: relative;
    border-radius: var(--radius-lg);
    background: var(--color-glass);
}}

.gradient-border::before {{
    content: '';
    position: absolute;
    inset: -1px;
    border-radius: inherit;
    padding: 1px;
    background: conic-gradient(from var(--angle, 0deg),
        var(--color-primary), var(--color-accent), var(--color-emerald), var(--color-primary));
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
    animation: rotateBorder 5s linear infinite;
    opacity: 0.6;
}}

@property --angle {{
    syntax: '<angle>';
    initial-value: 0deg;
    inherits: false;
}}

@keyframes rotateBorder {{
    to {{ --angle: 360deg; }}
}}

/* ---------- Avatars ---------- */

.avatar-circle {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px; height: 28px;
    border-radius: 50%;
    font-size: 14px;
    flex-shrink: 0;
}}

.avatar-user {{
    background: var(--color-surface-elevated);
    border: 1px solid var(--color-border-strong);
}}

.avatar-assistant {{
    background: var(--gradient-primary);
    box-shadow: var(--shadow-glow);
}}

/* ---------- Sidebar nav items ---------- */

.nav-section-label {{
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--color-text-muted);
    margin: var(--space-md) 0 var(--space-xs) 2px;
}}

.nav-item {{
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 10px;
    border-radius: var(--radius-sm);
    border-left: 2px solid transparent;
    color: var(--color-text-secondary);
    font-size: 13px;
    transition: all 0.15s ease;
}}

.nav-item:hover {{
    background: var(--color-surface-elevated);
    color: var(--color-text-primary);
}}

.nav-item.active {{
    border-left: 2px solid var(--color-accent);
    background: rgba(0, 212, 255, 0.08);
    color: var(--color-text-primary);
}}

/* ---------- Router timeline (native <details>, not st.expander) ---------- */

details.router-timeline {{
    background: var(--color-glass);
    backdrop-filter: blur(12px);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-lg);
    padding: 14px 16px;
    margin-top: 8px;
}}

details.router-timeline summary {{
    cursor: pointer;
    font-weight: 700;
    font-size: 13px;
    color: var(--color-text-primary);
    list-style: none;
}}

details.router-timeline summary::-webkit-details-marker {{
    display: none;
}}

.timeline-track {{
    position: relative;
    margin: 14px 0 4px 10px;
    padding-left: 20px;
    border-left: 2px solid transparent;
    border-image: linear-gradient(var(--color-primary), var(--color-accent), var(--color-emerald)) 1 100%;
}}

.timeline-node {{
    position: relative;
    padding-bottom: 16px;
    opacity: 0;
    animation: fadeInUp 0.4s ease both;
    transition: transform 0.15s ease;
}}

.timeline-node:hover {{
    transform: translateX(3px);
}}

.timeline-node::before {{
    content: '';
    position: absolute;
    left: -25px;
    top: 3px;
    width: 9px; height: 9px;
    border-radius: 50%;
    background: var(--color-accent);
    box-shadow: 0 0 6px var(--color-accent), 0 0 14px var(--color-accent);
    animation: glow 2s ease-in-out infinite;
}}

.timeline-node:last-child::before {{
    background: var(--color-emerald);
    box-shadow: 0 0 6px var(--color-emerald), 0 0 14px var(--color-emerald);
}}

/* ---------- Diagnostics dashboard ---------- */

.diag-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
    gap: 10px;
    margin-top: 8px;
}}

/* ---------- Responsive ---------- */

@media (max-width: 768px) {{
    .aurora-blob {{ width: 70vw; height: 70vw; }}
    .block-container {{ padding-left: 0.75rem !important; padding-right: 0.75rem !important; }}
}}
</style>
""",
        unsafe_allow_html=True,
    )
