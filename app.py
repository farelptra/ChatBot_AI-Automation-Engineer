import torch
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Fallback ke st.secrets jika dideploy di Streamlit Community Cloud
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

from rag_chatbot import (
    KNOWLEDGE_DIR,
    SYSTEM_PROMPT_PATH,
    TOP_K,
    buat_model,
    muat_dokumen,
    bangun_vectorstore,
    muat_system_prompt,
    buat_rag_chain,
)

# ============================================================
# 1. PENGATURAN HALAMAN
# ============================================================

st.set_page_config(
    page_title="SKRIPSI AI – Asisten Akademik Unindra",
    page_icon="🎓",
    layout="wide",
)

# ============================================================
# 2. CUSTOM CSS – TAMPILAN PREMIUM
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

/* ── Root Variables ── */
:root {
    --primary:        #6C63FF;
    --primary-light:  #8B85FF;
    --primary-dark:   #4A42D6;
    --accent:         #FF6B9D;
    --accent2:        #43E8C7;
    --bg-dark:        #0F0F1A;
    --bg-card:        #16162A;
    --bg-glass:       rgba(108, 99, 255, 0.08);
    --border-glass:   rgba(108, 99, 255, 0.20);
    --text-primary:   #F0EFFF;
    --text-secondary: #A8A5CC;
    --text-muted:     #6B68A0;
    --shadow-glow:    0 0 40px rgba(108, 99, 255, 0.25);
    --radius-lg:      20px;
    --radius-md:      14px;
    --radius-sm:      8px;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: var(--bg-dark) !important;
    color: var(--text-primary) !important;
}

.main .block-container {
    padding: 1.5rem 2rem 2rem !important;
    max-width: 900px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg-dark); }
::-webkit-scrollbar-thumb { background: var(--primary); border-radius: 99px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111128 0%, #0c0c1e 100%) !important;
    border-right: 1px solid var(--border-glass) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1.1rem 1.4rem !important;
    margin-bottom: 0.6rem !important;
    backdrop-filter: blur(10px);
    animation: fadeSlideIn 0.3s ease;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    border-color: rgba(108, 99, 255, 0.35) !important;
    background: linear-gradient(135deg, rgba(108,99,255,0.08) 0%, rgba(16,16,36,1) 100%) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    border-color: rgba(255,107,157,0.30) !important;
    background: linear-gradient(135deg, rgba(255,107,157,0.07) 0%, rgba(16,16,36,1) 100%) !important;
}

/* ── Chat Input ── */
[data-testid="stChatInput"] {
    background: #FFFFFF !important;
    border: 1.5px solid rgba(108, 99, 255, 0.35) !important;
    border-radius: 14px !important;
    padding: 0.2rem 0.5rem !important;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.06) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(108, 99, 255, 0.25) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #000000 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    caret-color: #000000 !important;
    font-size: 0.95rem !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #6B7280 !important;
}
/* Pastikan seluruh teks ketikan dan wrapper di dalam chat input berwarna hitam */
[data-testid="stChatInput"] input,
[data-testid="stChatInput"] p,
[data-testid="stChatInput"] span,
[data-testid="stChatInput"] div {
    color: #000000 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(108,99,255,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 25px rgba(108,99,255,0.45) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] > div {
    border-top-color: var(--primary) !important;
}

/* ── Markdown text inside chat ── */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
    font-size: 0.95rem;
    line-height: 1.7;
    color: var(--text-primary) !important;
}
[data-testid="stChatMessage"] code {
    background: rgba(108,99,255,0.15) !important;
    border: 1px solid rgba(108,99,255,0.25) !important;
    border-radius: 5px;
    padding: 2px 6px;
    font-size: 0.85rem;
}

/* ── Metric / Info cards in sidebar ── */
.stat-card {
    background: var(--bg-glass);
    border: 1px solid var(--border-glass);
    border-radius: var(--radius-md);
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
}
.stat-card .label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin-bottom: 2px;
}
.stat-card .value {
    font-size: 0.92rem;
    font-weight: 600;
    color: var(--primary-light);
}

/* ── Animations ── */
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulseGlow {
    0%, 100% { text-shadow: 0 0 20px rgba(108,99,255,0.5); }
    50%       { text-shadow: 0 0 40px rgba(108,99,255,0.9); }
}
@keyframes gradientShift {
    0%   { background-position: 0%   50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0%   50%; }
}

/* ── Header Hero ── */
.hero-container {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    position: relative;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(108,99,255,0.20), rgba(67,232,199,0.15));
    border: 1px solid rgba(108,99,255,0.35);
    border-radius: 99px;
    padding: 5px 16px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--accent2);
    margin-bottom: 0.8rem;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6C63FF 0%, #FF6B9D 50%, #43E8C7 100%);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradientShift 4s ease infinite;
    line-height: 1.15;
    margin-bottom: 0.5rem;
}
.hero-subtitle {
    font-size: 1rem;
    color: var(--text-secondary);
    font-weight: 400;
    max-width: 500px;
    margin: 0 auto 0.5rem;
    line-height: 1.6;
}
.hero-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-glass), transparent);
    margin: 1.5rem 0;
}

/* ── Quick suggestion pills ── */
.suggestion-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    margin: 0.5rem 0 1.5rem;
}
.suggestion-pill {
    background: var(--bg-glass);
    border: 1px solid var(--border-glass);
    border-radius: 99px;
    padding: 6px 14px;
    font-size: 0.8rem;
    color: var(--text-secondary);
    cursor: default;
    transition: all 0.2s;
}

/* ── Caption / label overrides ── */
.stCaption { color: var(--text-muted) !important; font-size: 0.78rem !important; }
p { color: var(--text-primary) !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 3. SIAPKAN MESIN CHATBOT
# ============================================================

@st.cache_resource(show_spinner="🎓 Memuat panduan skripsi & menyiapkan AI...")
def siapkan_chatbot():
    model = buat_model()
    dokumen = muat_dokumen(KNOWLEDGE_DIR)
    vectorstore = bangun_vectorstore(dokumen)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    system_prompt = muat_system_prompt(SYSTEM_PROMPT_PATH)
    return buat_rag_chain(retriever, model, system_prompt)

rag_chain = siapkan_chatbot()

# ============================================================
# 4. SESSION STATE
# ============================================================

if "riwayat" not in st.session_state:
    st.session_state.riwayat = []

# ============================================================
# 5. SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1.2rem 0 0.5rem;">
        <div style="font-size:3rem; margin-bottom:0.3rem;">🎓</div>
        <div style="font-size:1rem; font-weight:700; color:#F0EFFF;">SKRIPSI AI</div>
        <div style="font-size:0.72rem; color:#6B68A0; margin-top:2px;">Asisten Akademik Unindra</div>
    </div>
    <div style="height:1px; background:linear-gradient(90deg, transparent, rgba(108,99,255,0.3), transparent); margin: 1rem 0;"></div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-card">
        <div class="label">📚 Sumber Dokumen</div>
        <div class="value">Panduan Skripsi + Tugas Akhir</div>
    </div>
    <div class="stat-card">
        <div class="label">🏫 Institusi</div>
        <div class="value">Universitas Indraprasta PGRI</div>
    </div>
    <div class="stat-card">
        <div class="label">⚡ Model AI</div>
        <div class="value">GPT-OSS 120B via Groq</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1px; background:linear-gradient(90deg, transparent, rgba(108,99,255,0.3), transparent); margin: 1rem 0;'></div>", unsafe_allow_html=True)

    st.markdown("**💡 Contoh pertanyaan:**", help="Klik salin lalu tempel di kotak chat")
    contoh = [
        "Apa saja bab yang wajib ada di skripsi?",
        "Berapa halaman minimal untuk skripsi?",
        "Bagaimana format daftar pustaka yang benar?",
        "Apa persyaratan sidang skripsi?",
        "Bagaimana cara menulis abstrak yang baik?",
    ]
    for c in contoh:
        st.markdown(f"<div class='suggestion-pill' style='margin-bottom:4px; border-radius:8px;'>📌 {c}</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:1px; background:linear-gradient(90deg, transparent, rgba(108,99,255,0.3), transparent); margin: 1rem 0;'></div>", unsafe_allow_html=True)

    if st.button("🗑️ Reset Percakapan", use_container_width=True):
        st.session_state.riwayat = []
        st.rerun()

    st.markdown("""
    <div style="text-align:center; margin-top:1rem;">
        <div style="font-size:0.7rem; color:#6B68A0;">Jawaban berdasarkan dokumen panduan resmi.<br>Selalu konfirmasi ke prodi untuk keputusan akhir.</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 6. HERO HEADER
# ============================================================

st.markdown("""
<div class="hero-container">
    <div class="hero-badge">✦ Powered by RAG + Groq AI ✦</div>
    <div class="hero-title">SKRIPSI AI</div>
    <div class="hero-subtitle">
        Asisten akademik pintarmu untuk panduan penulisan<br>
        <strong style="color:#8B85FF;">Skripsi & Tugas Akhir Unindra</strong> 🎓
    </div>
    <div style="margin-top:0.8rem; display:flex; justify-content:center; gap:12px; flex-wrap:wrap;">
        <span style="background:rgba(108,99,255,0.12); border:1px solid rgba(108,99,255,0.25); border-radius:99px; padding:4px 12px; font-size:0.75rem; color:#A8A5CC;">📝 Format Penulisan</span>
        <span style="background:rgba(255,107,157,0.10); border:1px solid rgba(255,107,157,0.25); border-radius:99px; padding:4px 12px; font-size:0.75rem; color:#A8A5CC;">📚 Struktur Bab</span>
        <span style="background:rgba(67,232,199,0.10); border:1px solid rgba(67,232,199,0.25); border-radius:99px; padding:4px 12px; font-size:0.75rem; color:#A8A5CC;">🔬 Metodologi</span>
        <span style="background:rgba(108,99,255,0.12); border:1px solid rgba(108,99,255,0.25); border-radius:99px; padding:4px 12px; font-size:0.75rem; color:#A8A5CC;">📖 Daftar Pustaka</span>
    </div>
</div>
<div class="hero-divider"></div>
""", unsafe_allow_html=True)

# ============================================================
# 7. AREA PERCAKAPAN
# ============================================================

# Pesan selamat datang
with st.chat_message("assistant", avatar="🎓"):
    st.markdown(
        "Halo, Mahasiswa! 👋 Aku **SKRIPSI AI**, asisten akademikmu untuk semua pertanyaan "
        "seputar panduan penulisan skripsi dan tugas akhir **Universitas Indraprasta PGRI (Unindra)**. \n\n"
        "Silakan tanyakan apa saja — mulai dari format penulisan, struktur bab, persyaratan sidang, "
        "sampai cara menyusun daftar pustaka yang benar. Aku siap membantu! 📝✨"
    )

# Tampilkan riwayat
for pesan in st.session_state.riwayat:
    avatar = "🎓" if pesan["role"] == "assistant" else "👤"
    with st.chat_message(pesan["role"], avatar=avatar):
        st.markdown(pesan["isi"])

# ============================================================
# 8. INPUT & TANYA JAWAB
# ============================================================

pertanyaan = st.chat_input("Tanyakan seputar skripsi / tugas akhir Unindra... 📝")

if pertanyaan:
    with st.chat_message("user", avatar="👤"):
        st.markdown(pertanyaan)
    st.session_state.riwayat.append({"role": "user", "isi": pertanyaan})

    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("🔍 Mencari di panduan skripsi..."):
            jawaban = st.write_stream(rag_chain.stream(pertanyaan))
    st.session_state.riwayat.append({"role": "assistant", "isi": jawaban})