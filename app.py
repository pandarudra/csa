"""Interactive demo UI for the SpotifyCares support agent ("Harmony").

Streamlit chosen over a hand-rolled Flask+HTML app: one file, no JS to
write, and it's the standard low-effort choice for an internal ML demo like
this. Run with `streamlit run app.py`.

Custom font (Noto Sans JP, assets/Noto_Sans_JP/) is served via Streamlit's
built-in static file serving (`static/`, enabled in .streamlit/config.toml)
rather than hand-rolling a file server or inlining it as a multi-MB base64
data URI. `static/*.ttf` are real copies of the two weights actually used
(Regular, SemiBold), not symlinks into assets/ -- Streamlit's static
handler rejects symlinked files with a 400 (verified empirically), so a
copy is the only version of this that actually works.
"""
from __future__ import annotations

import time

import streamlit as st

from src.pipeline import SupportAgent
from src.schemas import AgentAction

st.set_page_config(page_title="Harmony", page_icon="assets/harmony.png", layout="centered")

st.markdown(
    """
    <style>
    @font-face {
        font-family: 'Noto Sans JP';
        src: url('app/static/NotoSansJP-Regular.ttf') format('truetype');
        font-weight: 400;
        font-display: swap;
    }
    @font-face {
        font-family: 'Noto Sans JP';
        src: url('app/static/NotoSansJP-SemiBold.ttf') format('truetype');
        font-weight: 600;
        font-display: swap;
    }
    html, body, [class*="css"] {
        font-family: 'Noto Sans JP', sans-serif;
    }
    h1, h2, h3, [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        font-weight: 600 !important;
    }
    .block-container {
        max-width: 700px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }
    #MainMenu, footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

EXAMPLES = {
    "Playback issue": "I can't play any songs on Premium, it just spins forever",
    "Double charge": "I was charged twice this month for my subscription, can I get one refund back?",
    "Can't log in": "Why can't I log in anymore? It says my password is wrong but I haven't changed it",
    "Feature request": "Any chance the new Drake album gets added soon?",
    "General complaint": "This app used to be great, now it crashes every time I open it. So frustrating.",
}


@st.cache_resource(show_spinner="Loading agent (intents, retrieval index, LLM client)...")
def load_agent() -> SupportAgent:
    return SupportAgent.load()


def _apply_example() -> None:
    choice = st.session_state.get("example_pill")
    if choice:
        st.session_state["message"] = EXAMPLES[choice]


st.title("Harmony")
st.caption(
    "A SpotifyCares support agent: classifies intent, drafts a reply grounded in real "
    "historical resolutions, and decides auto-handle vs. escalate. See report/REPORT.md "
    "for how well this is actually measured to work."
)

try:
    agent = load_agent()
except Exception as exc:  # noqa: BLE001 -- surfaced to the demo user, not swallowed
    st.error(f"Couldn't load the agent: {exc}\n\nIs NVIDIA_API_KEY set in .env?")
    st.stop()

st.pills("Try an example", list(EXAMPLES.keys()), key="example_pill", on_change=_apply_example)
message = st.text_area(
    "Customer message", key="message", height=100, placeholder="Type a tweet-style support message..."
)
run = st.button("Run agent", type="primary", disabled=not message.strip())

if run:
    start = time.monotonic()
    with st.spinner("Classifying, retrieving grounding, drafting reply, deciding escalation..."):
        result = agent.handle(message)
    elapsed = time.monotonic() - start

    with st.container(border=True):
        top = st.columns([2, 1])
        top[0].metric("Intent", result.intent, delta=f"{result.intent_confidence:.0%} confidence")
        with top[1]:
            st.markdown("<br>", unsafe_allow_html=True)
            if result.action == AgentAction.AUTO_HANDLE:
                st.success(f"auto_handle · {elapsed:.1f}s")
            else:
                st.warning(f"escalate · {elapsed:.1f}s")
        st.caption(f"Escalation reason: {result.escalation_reason}")

        st.markdown("**Drafted reply**")
        st.info(result.reply)

    with st.expander(f"Historical grounding used ({len(result.retrieved_examples)} cases retrieved)"):
        if not result.retrieved_examples:
            st.write("No closely similar historical case was found.")
        for example in result.retrieved_examples:
            st.markdown(f"`similarity={example.similarity:.2f}` `{example.intent}`")
            st.write(f"Customer: {example.customer_problem}")
            st.write(f"SpotifyCares: {example.brand_reply}")
            st.markdown("---")
