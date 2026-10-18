"""
LEVEL 2 · TASK 5 — Autocomplete & Autocorrect Data Analytics (NLP)
Run:  streamlit run app.py
"""
import random
import re
import time
from collections import Counter, defaultdict

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="TypeSmart NLP", page_icon="⌨️", layout="wide")
T = "plotly_dark"
LET = "abcdefghijklmnopqrstuvwxyz"

st.markdown(
    """<style>
.hero{padding:2rem 2.2rem;border-radius:20px;color:#fff;margin-bottom:1.2rem;
background:linear-gradient(135deg,#f59e0b 0%,#ec4899 50%,#8b5cf6 100%);box-shadow:0 10px 30px rgba(236,72,153,.35)}
.hero h1{margin:0;font-size:2.2rem;color:#fff}.hero p{margin:.4rem 0 0;opacity:.92}
div[data-testid="stMetric"]{background:rgba(127,127,127,.08);border:1px solid rgba(127,127,127,.22);
padding:14px 16px;border-radius:14px}
.chip{display:inline-block;padding:.35rem .9rem;margin:.2rem;border-radius:999px;font-weight:600;
background:linear-gradient(90deg,#6366f1,#ec4899);color:#fff}
</style>""",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero"><h1>⌨️ TypeSmart — Autocomplete & Autocorrect</h1><p>Frequency-based n-gram prediction and '
    "edit-distance spelling correction, built from scratch and benchmarked with precision / recall.</p></div>",
    unsafe_allow_html=True,
)

STOP = set("""a about above after again against all am an and any are as at be because been before being below between both but by
can did do does doing down during each few for from further had has have having he her here hers herself him himself his how i if in
into is it its itself just me more most my myself no nor not now of off on once only or other our ours ourselves out over own same
she should so some such than that the their theirs them themselves then there these they this those through to too under until up
very was we were what when where which while who whom why will with you your yours yourself yourselves would could shall may might
must upon unto thou thee thy thine hath doth""".split())

SAMPLE = """The quick brown fox jumps over the lazy dog. I am going to the market to buy some fresh bread and milk.
She said that the weather would be good tomorrow, and the children were happy to go to the park. In the morning the sun rose over the hills
and the village woke up. He was a kind man who loved to read books in the evening. It is a truth that every student must learn to think
clearly. There was a time when the town was quiet and the river was clean. They went to the school by the road and talked about the news.
I think that you should go to the doctor if you feel sick. We are going to the party and we will be there on time. The teacher gave the
class a new book and the students began to read it. She was going to the station to meet her friend. He said that the time had come to go home."""

DEFAULT_TESTS = """the_
i am_
in the_
of the_
he was_
it is_
there was_
and the_
she said_
to be_
i am go"""

MISSPELLED = [("recieve", "receive"), ("definately", "definitely"), ("seperate", "separate"), ("occured", "occurred"),
              ("untill", "until"), ("wich", "which"), ("becuase", "because"), ("freind", "friend"), ("beleive", "believe"),
              ("tomorow", "tomorrow"), ("wierd", "weird"), ("goverment", "government"), ("adress", "address"),
              ("accomodate", "accommodate"), ("enviroment", "environment"), ("thier", "their"), ("calender", "calendar"),
              ("embarass", "embarrass"), ("neccessary", "necessary"), ("publically", "publicly"), ("teh", "the"),
              ("adn", "and"), ("hapen", "happen"), ("speach", "speech"), ("langauge", "language")]


# ───────────────────────── corpus & preprocessing ─────────────────────────
def tokenize(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


@st.cache_data(show_spinner="Loading Project Gutenberg books via NLTK…")
def load_gutenberg(books):
    import nltk
    try:
        nltk.data.find("corpora/gutenberg")
    except LookupError:
        nltk.download("gutenberg", quiet=True)
    from nltk.corpus import gutenberg
    return " ".join(gutenberg.raw(b) for b in books)


with st.sidebar:
    st.header("⚙️ Corpus")
    src = st.radio("Text source", ["Project Gutenberg (NLTK)", "Upload .txt", "Tiny built-in sample"])
    text, label = SAMPLE * 3, "built-in sample"
    if src.startswith("Project"):
        books = st.multiselect("Books", ["austen-emma.txt", "austen-persuasion.txt", "austen-sense.txt", "carroll-alice.txt",
                                         "melville-moby_dick.txt", "chesterton-thursday.txt", "bryant-stories.txt", "shakespeare-hamlet.txt"],
                               default=["austen-emma.txt", "austen-persuasion.txt", "austen-sense.txt", "carroll-alice.txt"])
        try:
            text, label = load_gutenberg(tuple(books)), "Gutenberg: " + ", ".join(b.replace(".txt", "") for b in books)
        except Exception as e:  # offline etc.
            st.warning(f"Could not download corpus ({type(e).__name__}). Using the tiny sample instead.")
    elif src.startswith("Upload"):
        up = st.file_uploader("Plain-text file", type="txt")
        if up:
            text, label = up.read().decode("utf-8", errors="ignore"), up.name
        else:
            st.info("Upload a .txt file (e.g. a Gutenberg book) — using the sample meanwhile.")
    min_freq = st.slider("Min word frequency for dictionary", 1, 10, 2)
    n_eval = st.slider("Autocomplete test positions", 300, 5000, 1500, 100)
    st.caption(f"Corpus: {label}")

tokens = tokenize(text)
if len(tokens) < 500:
    st.error("Corpus too small – add more text.")
    st.stop()

# chunked 80/20 split so train & test come from the same mix of text
chunks = [tokens[i:i + 500] for i in range(0, len(tokens), 500)]
random.Random(42).shuffle(chunks)
cut = int(len(chunks) * .8)
train_tok = [t for c in chunks[:cut] for t in c]
test_chunks = chunks[cut:]
KEY = f"{label}|{len(tokens)}"


@st.cache_resource(show_spinner="Building n-gram models…")
def build(key, _train):
    uni, bi, tri = Counter(_train), defaultdict(Counter), defaultdict(Counter)
    for a, b in zip(_train, _train[1:]):
        bi[a][b] += 1
    for a, b, c in zip(_train, _train[1:], _train[2:]):
        tri[(a, b)][c] += 1
    return dict(uni=uni, bi=bi, tri=tri, ranked=[w for w, _ in uni.most_common()])


M = build(KEY, train_tok)


def predict(ctx, partial="", k=3, order=3):
    """order 1 = unigram frequency, 2 = bigram (+backoff), 3 = trigram (+backoff)."""
    out, srcs = [], []
    if order >= 3 and len(ctx) >= 2:
        srcs.append(M["tri"].get((ctx[-2], ctx[-1])))
    if order >= 2 and len(ctx) >= 1:
        srcs.append(M["bi"].get(ctx[-1]))
    for c in srcs:
        if c:
            for w, _ in c.most_common():
                if w.startswith(partial) and w not in out:
                    out.append(w)
                    if len(out) == k:
                        return out
    for w in M["ranked"]:
        if w.startswith(partial) and w not in out:
            out.append(w)
            if len(out) == k:
                break
    return out


def suggest_text(s, order=3, k=3):
    toks = tokenize(s)
    if not toks:
        return predict([], "", k, order)
    if s.endswith(" "):
        return predict(toks, "", k, order)
    return predict(toks[:-1], toks[-1], k, order)


# ───────────────────────── spell-correction algorithms ─────────────────────────
@st.cache_resource(show_spinner="Building dictionary…")
def build_dict(key, mf, _uni):
    freq = {w: c for w, c in _uni.items() if c >= mf and w.isalpha()}
    by_len = defaultdict(list)
    for w in freq:
        by_len[len(w)].append(w)
    return freq, by_len


FREQ, BY_LEN = build_dict(KEY, min_freq, M["uni"])


def lev(a, b, maxd=2):
    if abs(len(a) - len(b)) > maxd:
        return maxd + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > maxd:
            return maxd + 1
        prev = cur
    return prev[-1]


def correct_lev(w):
    if w in FREQ:
        return w
    best = None
    for L in range(len(w) - 2, len(w) + 3):
        for cand in BY_LEN.get(L, []):
            d = lev(w, cand)
            if d <= 2 and (best is None or (d, -FREQ[cand]) < best[0]):
                best = ((d, -FREQ[cand]), cand)
    return best[1] if best else w


def edits1(w):
    sp = [(w[:i], w[i:]) for i in range(len(w) + 1)]
    return ({a + b[1:] for a, b in sp if b} | {a + b[1] + b[0] + b[2:] for a, b in sp if len(b) > 1}
            | {a + c + b[1:] for a, b in sp if b for c in LET} | {a + c + b for a, b in sp for c in LET})


def correct_norvig(w):
    if w in FREQ:
        return w
    e1 = [x for x in edits1(w) if x in FREQ]
    if e1:
        return max(e1, key=FREQ.get)
    e2 = {y for x in edits1(w) for y in edits1(x) if y in FREQ}
    return max(e2, key=FREQ.get) if e2 else w


try:
    from spellchecker import SpellChecker
    _sp = SpellChecker()
    def correct_pysp(w):
        return _sp.correction(w) or w
    HAS_PYSP = True
except Exception:
    HAS_PYSP = False

ALGOS = {"Levenshtein (custom DP)": correct_lev, "Norvig edits (Damerau-style)": correct_norvig}
if HAS_PYSP:
    ALGOS["pyspellchecker"] = correct_pysp


@st.cache_data(show_spinner="Benchmarking autocorrect algorithms…")
def eval_correct(key, mf, pairs_key, _pairs, _controls):
    out = {}
    for name, fn in ALGOS.items():
        t0 = time.perf_counter()
        outs = [fn(w) for w, _ in _pairs]
        fixed = [o == r for o, (_, r) in zip(outs, _pairs)]
        wrong_fix = [(w, o) for o, (w, r) in zip(outs, _pairs) if o != r]
        changed_ctrl = [c for c in _controls if fn(c) != c]
        dt = (time.perf_counter() - t0) / (len(_pairs) + len(_controls))
        tp = sum(fixed)
        n_changes = sum(o != w for o, (w, _) in zip(outs, _pairs)) + len(changed_ctrl)  # words the algorithm changed
        out[name] = dict(tp=tp, miss=len(_pairs) - tp, false_fix=len(changed_ctrl), ok_ctrl=len(_controls) - len(changed_ctrl),
                         precision=tp / n_changes if n_changes else 0, recall=tp / len(_pairs),
                         accuracy=(tp + len(_controls) - len(changed_ctrl)) / (len(_pairs) + len(_controls)), ms=dt * 1000,
                         wrong=wrong_fix)
    return out


# ───────────────────────── autocomplete evaluation ─────────────────────────
@st.cache_data(show_spinner="Evaluating autocomplete models…")
def eval_complete(key, n):
    pairs = [(c[i - 2:i], c[i]) for c in test_chunks for i in range(2, len(c))]
    random.Random(1).shuffle(pairs)
    pairs = pairs[:n]
    rows = []
    for order, name in [(1, "Unigram frequency"), (2, "Bigram"), (3, "Trigram + backoff")]:
        for mode in ["Next word", "Word completion (2 typed letters)"]:
            hit1 = hit3 = shown = tot = 0
            t0 = time.perf_counter()
            for ctx, true in pairs:
                part = ""
                if mode.startswith("Word"):
                    if len(true) < 3:
                        continue
                    part = true[:2]
                p = predict(ctx, part, 3, order)
                tot += 1
                shown += len(p)
                hit1 += bool(p) and p[0] == true
                hit3 += true in p
            ms = (time.perf_counter() - t0) / max(tot, 1) * 1000
            rows.append(dict(Model=name, Task=mode, **{"Top-1 accuracy": hit1 / tot, "Recall@3": hit3 / tot,
                                                        "Precision@3": hit3 / shown if shown else 0, "ms / query": ms}))
    return pd.DataFrame(rows)


tabs = st.tabs(["🧹 Corpus & Preprocessing", "🔮 Autocomplete", "✏️ Autocorrect", "📊 Metrics & Comparison", "🎮 Playground", "📝 Discussion"])

# ───────────────────────── tab 1 ─────────────────────────
with tabs[0]:
    nostop = [t for t in tokens if t not in STOP and len(t) > 1]
    c = st.columns(4)
    c[0].metric("Tokens", f"{len(tokens):,}")
    c[1].metric("Vocabulary", f"{len(set(tokens)):,}")
    c[2].metric("Train / test tokens", f"{len(train_tok):,} / {len(tokens) - len(train_tok):,}")
    c[3].metric("After stopword removal", f"{len(nostop):,}")
    st.subheader("Preprocessing pipeline (first sentence)")
    sent = re.split(r"(?<=[.!?])\s+", text.strip())[0][:200].replace("\n", " ")
    low = sent.lower()
    nop = re.sub(r"[^\w\s']", "", low)
    tok = tokenize(sent)
    st.table(pd.DataFrame({"step": ["1. raw", "2. lowercase", "3. punctuation removed", "4. tokenised", "5. stopwords removed"],
                           "result": [sent, low, nop, str(tok), str([t for t in tok if t not in STOP])]}))
    st.subheader("Top 20 most frequent words (stopwords removed)")
    top = pd.DataFrame(Counter(nostop).most_common(20), columns=["word", "count"])
    st.plotly_chart(px.bar(top, x="count", y="word", orientation="h", template=T, color="count", color_continuous_scale="Sunsetdark")
                    .update_layout(yaxis=dict(autorange="reversed"), height=560, coloraxis_showscale=False), use_container_width=True)
    st.caption("Stopwords are removed for frequency analysis only. For next-word prediction we keep them — "
               "“of the”, “to be” are exactly what a keyboard must predict.")

# ───────────────────────── tab 2 ─────────────────────────
with tabs[1]:
    st.subheader("Frequency-based n-gram autocomplete — 10+ test inputs, top-3 predictions")
    st.caption("Input ending with a **space** → predict the next word. Input ending with letters → complete the partial word.")
    lines = [l for l in st.text_area("Test inputs (one per line; a trailing `_` means a space → predict the next word)", DEFAULT_TESTS, height=240).splitlines() if l.strip()]
    rows = []
    for l in lines:
        s = l[:-1] + " " if l.endswith("_") else l
        rows.append({"Input": s.replace(" ", "·") if s.endswith(" ") else s,
                     "Unigram": ", ".join(suggest_text(s, 1)), "Bigram": ", ".join(suggest_text(s, 2)),
                     "Trigram": ", ".join(suggest_text(s, 3))})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(f"{len(rows)} inputs tested · “·” marks a trailing space (next-word prediction).")

# ───────────────────────── tab 3 ─────────────────────────
with tabs[2]:
    st.subheader("Autocorrect on deliberately misspelled words")
    st.caption(f"Dictionary: {len(FREQ):,} words from the corpus (frequency ≥ {min_freq})."
               + (" pyspellchecker also uses its own built-in dictionary." if HAS_PYSP else " Install `pyspellchecker` to add a third approach."))
    df_pairs = st.data_editor(pd.DataFrame(MISSPELLED, columns=["misspelled", "correct"]), num_rows="dynamic", use_container_width=True, height=300)
    pairs = [(a.strip().lower(), b.strip().lower()) for a, b in df_pairs.dropna().values if a and b]
    rng = random.Random(3)
    rare = [w for w, c in FREQ.items() if 2 <= c <= 30 and len(w) >= 5]
    controls = rng.sample(rare, min(25, len(rare)))
    ev = eval_correct(KEY, min_freq, str(pairs), pairs, controls)
    pick = st.selectbox("Algorithm", list(ev))
    r = ev[pick]
    detail = [{"misspelled": w, "expected": c, "corrected": ALGOS[pick](w), "✓": "✅" if ALGOS[pick](w) == c else "❌"} for w, c in pairs]
    st.dataframe(pd.DataFrame(detail), use_container_width=True, hide_index=True)
    st.metric("Correction accuracy on misspellings", f"{100 * r['recall']:.1f}%")

# ───────────────────────── tab 4 ─────────────────────────
with tabs[3]:
    st.subheader("Autocorrect — precision, recall, confusion matrix")
    st.markdown("""
- **Recall** = misspellings correctly fixed ÷ all misspellings
- **Precision** = misspellings correctly fixed ÷ all words the algorithm *changed* (wrong fixes and damage to correct words count against it)
- **Accuracy** = (fixed misspellings + correct words left alone) ÷ all test words  
Test set = your misspelled words **+ 25 rare-but-valid words** (to catch over-correction).
""")
    comp = pd.DataFrame({n: {"Precision": r["precision"], "Recall": r["recall"], "Accuracy": r["accuracy"], "ms / word": r["ms"]} for n, r in ev.items()}).T
    st.dataframe(comp.style.format("{:.3f}"), use_container_width=True)
    long = comp.drop(columns="ms / word").reset_index().melt("index", var_name="metric", value_name="score")
    a, b = st.columns(2)
    a.plotly_chart(px.bar(long, x="index", y="score", color="metric", barmode="group", template=T, text_auto=".2f", labels={"index": ""}),
                   use_container_width=True)
    b.plotly_chart(px.bar(comp.reset_index(), x="index", y="ms / word", template=T, color="index", log_y=True, labels={"index": ""},
                          title="Speed (ms per word, log scale)").update_layout(showlegend=False), use_container_width=True)
    name = st.selectbox("Confusion matrix for", list(ev), key="cm")
    r = ev[name]
    cm = pd.DataFrame([[r["ok_ctrl"], r["false_fix"]], [r["tp"], r["miss"]]],
                      index=["Before: correct word", "Before: misspelled"], columns=["After: correct", "After: incorrect"])
    st.plotly_chart(px.imshow(cm, text_auto=True, template=T, color_continuous_scale="Purples", title=f"Autocorrect before vs after — {name}"),
                    use_container_width=True)
    st.subheader("Autocomplete — benchmark on held-out text")
    st.markdown("**Top-1** = first suggestion is right · **Recall@3** = true word is in the top 3 · "
                "**Precision@3** = correct suggestions ÷ all suggestions shown.")
    res = eval_complete(KEY, n_eval)
    fmt = {c: "{:.3f}" for c in res.columns if c not in ("Model", "Task")}
    st.dataframe(res.style.format(fmt), use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(res, x="Model", y="Recall@3", color="Task", barmode="group", template=T, text_auto=".2f"), use_container_width=True)
    best = res[res.Task == "Next word"].sort_values("Recall@3").iloc[-1]
    st.success(f"Best next-word model: **{best.Model}** (Recall@3 = {best['Recall@3']:.1%}). "
               "Higher-order n-grams are more precise but see fewer contexts, hence the backoff to lower orders.")

# ───────────────────────── tab 5 ─────────────────────────
with tabs[4]:
    s = st.text_input("Type something…", "i am going to the")
    order = st.radio("Autocomplete model", [3, 2, 1], format_func={3: "Trigram", 2: "Bigram", 1: "Unigram"}.get, horizontal=True)
    sug = suggest_text(s, order)
    st.markdown("**Suggestions:** " + " ".join(f'<span class="chip">{w}</span>' for w in sug), unsafe_allow_html=True)
    algo = st.selectbox("Autocorrect algorithm", list(ALGOS), key="play_algo")
    words = s.split()
    fixed = [ALGOS[algo](w.lower()) if w.isalpha() else w for w in words]
    st.markdown(f"**Autocorrected:** {' '.join(fixed)}")
    if words and fixed != [w.lower() for w in words]:
        st.caption("Changed: " + ", ".join(f"{a} → {b}" for a, b in zip(words, fixed) if a.lower() != b))

# ───────────────────────── tab 6 ─────────────────────────
with tabs[5]:
    st.subheader("Limitations vs production keyboards (e.g. Google Gboard)")
    st.markdown("""
| Area | This project | Production (Gboard etc.) |
|---|---|---|
| **Language model** | Counts of 2–3 word sequences | Neural LMs (LSTM / Transformer) with long context and semantic understanding |
| **Data sparsity** | Unseen n-grams → backoff, rare words unknown | Subword tokens, embeddings generalise to unseen words |
| **Personalisation** | None | On-device learning from your own typing; contacts, slang, emoji; federated learning protects privacy |
| **Spelling errors** | Pure edit distance on text | Touch-point model: knows keyboard geometry, which neighbouring key you probably hit |
| **Real-word errors** | Cannot detect *their / there* | Context-aware correction using sentence meaning |
| **Corpus** | Old public-domain books → outdated vocabulary | Billions of modern messages, many languages, code-switching |
| **Ranking** | Frequency only | Learned ranking combining context, recency, user history, app type |
| **Latency & memory** | Python dicts, fine for a demo | Quantised models under ~10 ms and a few MB on-device |
| **Evaluation** | Offline accuracy | Online A/B tests on keystroke savings and acceptance rate |

**Take-aways**
1. Frequency n-grams are a strong, explainable baseline, but accuracy plateaus because they ignore meaning beyond 1–2 words.
2. Edit distance ≤ 2 fixes most typos, but ranking by raw frequency can pick the wrong candidate (e.g. *form* vs *from*); context fixes that.
3. Metrics depend heavily on corpus coverage — a bigger, in-domain corpus is the cheapest improvement.
4. Next steps: Kneser–Ney smoothing, a small neural LM, keyboard-adjacency weights, and personalised dictionaries.
""")
