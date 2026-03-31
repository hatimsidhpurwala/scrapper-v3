import streamlit as st
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import time
import requests
from bs4 import BeautifulSoup

st.set_page_config(
    page_title="Universal Web-Based AI System",
    page_icon="🧠",
    layout="centered"
)

st.markdown("""
<style>
.answer-box {
    border-left: 4px solid #7c6ef7;
    border-radius: 8px;
    padding: 20px 24px;
    margin-top: 12px;
    font-size: 15px;
    line-height: 1.9;
}
.step-card { border-radius: 8px; padding: 10px 16px; margin: 5px 0; font-size: 14px; }
.step-done   { color: #22c55e; }
.step-active { color: #f59e0b; }
.step-wait   { color: #6b7280; }
.source-box  { border-left: 3px solid #374151; padding: 8px 14px; margin: 8px 0;
               font-size: 13px; border-radius: 0 6px 6px 0; }
</style>
""", unsafe_allow_html=True)

for key in ["index","chunks","embed_model","ready","stats","raw_text_full"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "ready" not in st.session_state:
    st.session_state.ready = False


# ── Model loader ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# ══════════════════════════════════════════════════════════════════════════════
#  SCRAPING — Multi-strategy with full content extraction
# ══════════════════════════════════════════════════════════════════════════════

def scrape_with_requests(url):
    """Primary scraper — extracts full visible text, not just headings."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.text, resp.url


def extract_full_text(html, base_url=""):
    """
    Deep extraction — gets ALL visible text including content inside
    divs, spans, sections. Groups nearby text into coherent passages.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove purely non-content elements
    for tag in soup(["script","style","noscript","iframe","svg",
                     "meta","link","head","button","input","select",
                     "textarea","figure","picture"]):
        tag.decompose()

    page_title = ""
    if soup.title and soup.title.string:
        page_title = soup.title.string.strip()

    passages = []

    # ── Pass 1: Semantic HTML blocks ──────────────────────────────────────────
    # Grab article/section/main content first (highest signal)
    for container in soup.find_all(["article","main","section","[role=main]"]):
        text = container.get_text(separator=" ", strip=True)
        if len(text) > 100:
            passages.append(("semantic", text))

    # ── Pass 2: Heading + body pairs ─────────────────────────────────────────
    for heading in soup.find_all(["h1","h2","h3","h4","h5"]):
        h_text = heading.get_text(" ", strip=True)
        if not h_text or len(h_text) < 3:
            continue

        # Collect ALL text siblings until the next heading of same/higher level
        body_parts = []
        for sib in heading.find_next_siblings():
            if sib.name in ["h1","h2","h3","h4","h5"]:
                break
            # Get full text of this sibling including all nested elements
            sib_text = sib.get_text(separator=" ", strip=True)
            if sib_text and len(sib_text) > 15:
                body_parts.append(sib_text)

        if body_parts:
            full_block = h_text + ". " + " ".join(body_parts)
            passages.append(("heading_block", full_block))
        else:
            passages.append(("heading", h_text))

    # ── Pass 3: All paragraphs ────────────────────────────────────────────────
    for p in soup.find_all("p"):
        text = p.get_text(separator=" ", strip=True)
        if len(text) > 30:
            passages.append(("paragraph", text))

    # ── Pass 4: List items — joined with context ──────────────────────────────
    for ul in soup.find_all(["ul","ol"]):
        items = []
        for li in ul.find_all("li", recursive=False):
            item_text = li.get_text(separator=" ", strip=True)
            if item_text and len(item_text) > 5:
                items.append(item_text)
        if items:
            # Try to find a label/heading before the list
            prev = ul.find_previous(["h1","h2","h3","h4","p","strong","b"])
            label = prev.get_text(" ", strip=True) if prev else ""
            prefix = (label + ": ") if label and len(label) < 120 else ""
            joined = prefix + " | ".join(items)
            passages.append(("list", joined))

    # ── Pass 5: Tables ────────────────────────────────────────────────────────
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td","th"])]
            row = " | ".join(c for c in cells if c)
            if row:
                rows.append(row)
        if rows:
            passages.append(("table", " || ".join(rows)))

    # ── Pass 6: Div / span text blocks (catches JS-rendered text) ────────────
    for div in soup.find_all(["div","span","li","td"]):
        # Only direct text children to avoid duplication
        direct_parts = []
        for child in div.children:
            if hasattr(child, 'string') and child.string:
                t = child.string.strip()
                if t:
                    direct_parts.append(t)
            elif hasattr(child, 'get_text'):
                # Include if it's a leaf element (no block children)
                if not child.find(["div","p","h1","h2","h3","h4","ul","ol","table"]):
                    t = child.get_text(" ", strip=True)
                    if t:
                        direct_parts.append(t)
        combined = " ".join(direct_parts)
        if len(combined) > 80:
            passages.append(("div_text", combined))

    return [t for _, t in passages], page_title


# ── CLEANING ───────────────────────────────────────────────────────────────────
NOISE = {
    "cookie","privacy policy","terms of service","terms & conditions",
    "login","sign up","sign in","register now","subscribe","newsletter",
    "all rights reserved","click here","read more","learn more","skip to",
    "back to top","loading","please wait","javascript","enable javascript",
    "404","page not found","go back","home page"
}

def clean(texts):
    seen, out = set(), []
    for t in texts:
        t = re.sub(r'\s+', ' ', t).strip()
        # Remove non-ASCII junk
        t = re.sub(r'[^\x20-\x7E]', ' ', t)
        t = re.sub(r'\s+', ' ', t).strip()

        if len(t) < 35:
            continue

        low = t.lower()
        if any(n in low for n in NOISE):
            continue

        # Must be mostly alphabetic
        if sum(c.isalpha() for c in t) / len(t) < 0.45:
            continue

        # Dedup
        key = re.sub(r'\W+', '', low)[:120]
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


# ── CHUNKING ───────────────────────────────────────────────────────────────────
def chunk_texts(texts, max_words=180, overlap=40):
    chunks = []
    for t in texts:
        words = t.split()
        if len(words) < 10:
            continue
        if len(words) <= max_words:
            chunks.append(t)
        else:
            for i in range(0, len(words), max_words - overlap):
                part = words[i: i + max_words]
                if len(part) >= 15:
                    chunks.append(" ".join(part))
    return chunks


# ── FAISS ─────────────────────────────────────────────────────────────────────
def build_index(chunks, model):
    embs = model.encode(chunks, show_progress_bar=False, batch_size=32)
    embs = np.array(embs, dtype="float32")
    faiss.normalize_L2(embs)
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    return idx


def retrieve(query, idx, chunks, model, top_k=8):
    q = np.array(model.encode([query]), dtype="float32")
    faiss.normalize_L2(q)
    scores, ids = idx.search(q, top_k)
    return [(chunks[i], float(scores[0][j]))
            for j, i in enumerate(ids[0]) if i < len(chunks)]


# ══════════════════════════════════════════════════════════════════════════════
#  ANSWER GENERATION — Using Claude API (much better than Flan-T5)
# ══════════════════════════════════════════════════════════════════════════════

def generate_answer_claude(query, context_chunks, api_key):
    """Use Claude claude-sonnet-4-20250514 via API for high-quality answers."""
    # Build context from top chunks, up to ~600 words
    context_parts, word_count = [], 0
    for text, score in context_chunks:
        words = text.split()
        if word_count + len(words) > 600:
            break
        context_parts.append(text)
        word_count += len(words)

    context = "\n\n".join(context_parts)

    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 512,
        "messages": [{
            "role": "user",
            "content": (
                f"You are a helpful assistant. Using ONLY the information provided below, "
                f"write a clear, detailed paragraph that answers the question. "
                f"Do not add information not present in the context. "
                f"If the context does not contain enough information, say so.\n\n"
                f"CONTEXT:\n{context}\n\n"
                f"QUESTION: {query}\n\n"
                f"ANSWER (write as a clear paragraph):"
            )
        }]
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        json=payload, headers=headers, timeout=30
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"].strip()


def generate_answer_flan(query, context_chunks):
    """Fallback: Flan-T5-Large with improved prompt."""
    from transformers import T5ForConditionalGeneration, T5Tokenizer

    @st.cache_resource(show_spinner=False)
    def _load():
        tok = T5Tokenizer.from_pretrained("google/flan-t5-large")
        mdl = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
        return mdl, tok

    model, tok = _load()

    context_parts, wc = [], 0
    for text, _ in context_chunks:
        words = text.split()
        if wc + len(words) > 350:
            break
        context_parts.append(text)
        wc += len(words)
    context = " ".join(context_parts)

    prompt = (
        f"Read the following information carefully and answer the question in detail.\n\n"
        f"Information: {context}\n\n"
        f"Question: {query}\n\n"
        f"Detailed answer:"
    )
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=700)
    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        min_new_tokens=50,
        num_beams=5,
        no_repeat_ngram_size=3,
        length_penalty=2.0,
        early_stopping=True,
    )
    answer = tok.decode(outputs[0], skip_special_tokens=True)

    # Fallback if too short
    if len(answer.strip()) < 40:
        sentences = []
        for part in context_parts:
            for s in re.split(r'(?<=[.!?])\s+', part):
                if len(s.strip()) > 40:
                    sentences.append(s.strip())
        answer = " ".join(sentences[:5]) if sentences else "Could not generate an answer from the available content."

    return answer


# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════

st.title("🧠 Universal Web-Based AI System")
st.caption("Enter any website URL → scrape → clean → index → query with AI.")
st.divider()

# ── API Key (optional — enables Claude for better answers) ────────────────────
with st.expander("⚙️ Optional: Add Anthropic API key for better answers"):
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="If provided, uses Claude for answers. Otherwise uses Flan-T5 (free, less accurate)."
    )
    if api_key:
        st.success("Claude API key set — will use Claude for answers.")
    else:
        st.info("No API key — will use Flan-T5 (free, runs locally).")

st.subheader("Step 1 — Process a website")

col_url, col_btn = st.columns([4, 1])
with col_url:
    url = st.text_input("URL", placeholder="https://example.com",
                        label_visibility="collapsed")
with col_btn:
    process_btn = st.button("Process", use_container_width=True, type="primary")

progress_box = st.empty()

STEPS = [
    ("Fetching website HTML…",      "Website fetched"),
    ("Extracting all text…",        "Text extracted"),
    ("Cleaning & deduplicating…",   "Data cleaned"),
    ("Chunking into passages…",     "Passages ready"),
    ("Building vector index…",      "Vector index built"),
    ("System ready!",               "Ready — ask your question below"),
]

def render_steps(done, active):
    html = ""
    for i, (a_lbl, d_lbl) in enumerate(STEPS):
        if i < done:
            html += f'<div class="step-card step-done">✅ {d_lbl}</div>'
        elif i == active:
            html += f'<div class="step-card step-active">⏳ {a_lbl}</div>'
        else:
            html += f'<div class="step-card step-wait">○ {d_lbl}</div>'
    return html


if process_btn and url:
    st.session_state.ready = False
    try:
        # Step 0 — Fetch
        progress_box.markdown(render_steps(-1, 0), unsafe_allow_html=True)
        html_content, final_url = scrape_with_requests(url)

        # Step 1 — Extract
        progress_box.markdown(render_steps(1, 1), unsafe_allow_html=True)
        raw_texts, page_title = extract_full_text(html_content, final_url)

        if not raw_texts:
            st.error("No text found. The site may require JavaScript rendering. Try a different URL.")
            st.stop()

        # Step 2 — Clean
        progress_box.markdown(render_steps(2, 2), unsafe_allow_html=True)
        cleaned = clean(raw_texts)
        if not cleaned:
            st.error("All extracted text was filtered out. The site may have too little readable content.")
            st.stop()

        # Step 3 — Chunk
        progress_box.markdown(render_steps(3, 3), unsafe_allow_html=True)
        chunks = chunk_texts(cleaned)
        if not chunks:
            st.error("Could not create text chunks.")
            st.stop()

        # Step 4 — Embed + Index
        progress_box.markdown(render_steps(4, 4), unsafe_allow_html=True)
        embed_model = load_embed_model()
        index = build_index(chunks, embed_model)

        # Done
        progress_box.markdown(render_steps(6, -1), unsafe_allow_html=True)

        st.session_state.index       = index
        st.session_state.chunks      = chunks
        st.session_state.embed_model = embed_model
        st.session_state.ready       = True
        st.session_state.stats       = {
            "title":   page_title or url,
            "raw":     len(raw_texts),
            "cleaned": len(cleaned),
            "chunks":  len(chunks),
        }

    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP error: {e}")
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the URL: {e}")
    except Exception as e:
        st.error(f"Error: {e}")
        raise

elif process_btn:
    st.warning("Please enter a URL.")


if st.session_state.ready and st.session_state.stats:
    s = st.session_state.stats
    st.success(f"✅ **{s['title']}** is ready!")
    c1, c2, c3 = st.columns(3)
    c1.metric("Raw blocks",    s["raw"])
    c2.metric("Clean blocks",  s["cleaned"])
    c3.metric("Chunks indexed", s["chunks"])

st.divider()

# ── STEP 2: Query ──────────────────────────────────────────────────────────────
st.subheader("Step 2 — Ask a question")

if not st.session_state.ready:
    st.info("Process a website above first.")
else:
    query   = st.text_input("Your question",
                            placeholder="What services do you provide?")
    ask_btn = st.button("Get Answer", type="primary")

    if ask_btn and query:
        with st.spinner("Finding relevant content and generating answer…"):
            results = retrieve(
                query,
                st.session_state.index,
                st.session_state.chunks,
                st.session_state.embed_model,
                top_k=8,
            )

            # Check if best match score is too low
            best_score = results[0][1] if results else 0
            if best_score < 0.15:
                st.warning(
                    "⚠️ Low similarity scores — the website may not contain information "
                    "directly related to your question. Try rephrasing."
                )

            try:
                if api_key and api_key.startswith("sk-ant"):
                    answer = generate_answer_claude(query, results, api_key)
                    model_used = "Claude claude-sonnet-4-20250514"
                else:
                    answer = generate_answer_flan(query, results)
                    model_used = "Flan-T5-Large"
            except Exception as e:
                st.error(f"Answer generation failed: {e}")
                answer = None
                model_used = ""

        if answer:
            st.markdown("#### Answer")
            st.markdown(
                f'<div class="answer-box">{answer}</div>',
                unsafe_allow_html=True
            )
            st.caption(f"Generated by: {model_used}")

            with st.expander("View source passages used"):
                for i, (chunk_text, score) in enumerate(results[:5], 1):
                    st.markdown(f"**Source {i}** — similarity: `{score:.3f}`")
                    st.markdown(
                        f'<div class="source-box">{chunk_text}</div>',
                        unsafe_allow_html=True
                    )

    elif ask_btn:
        st.warning("Please enter a question.")

st.divider()
st.caption("Built with Streamlit · FAISS · SentenceTransformers · Claude / Flan-T5 · BeautifulSoup")
