"""
app.py - Dashboard Streamlit: Switching Hybrid Book Recommender (Book-Crossing)
Jalankan:  streamlit run app.py
Butuh   :  folder artifacts/ (hasil book_recommender.py) + recommender_core.py
"""
import json
import re
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from recommender_core import Recommender

st.set_page_config(page_title="Book Recommender", page_icon="📚", layout="wide")

ART = Path(__file__).parent / "artifacts"
if not (ART / "model_state.pkl").exists():
    st.error("Folder `artifacts/` belum ada. Jalankan dulu `python book_recommender.py` "
             "lalu salin folder `artifacts/` ke sebelah app.py.")
    st.stop()


@st.cache_resource(show_spinner="Memuat model ...")
def load_model():
    return Recommender(str(ART))


@st.cache_data
def load_dash():
    with open(ART / "dashboard_data.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_test_sample():
    return pd.read_csv(ART / "test_sample.csv")


rec = load_model()
D = load_dash()
CFG = rec.cfg
K = CFG["TOP_K"]

ROUTE_INFO = {
    "CF": ("success", "**Item-Based CF** — user punya cukup riwayat, rekomendasi dihitung dari kemiripan antar-buku."),
    "CF (profil baru)": ("success", "**Item-Based CF** dari profil yang kamu isi barusan."),
    "Content": ("info", "**Content-Based** — riwayat user terlalu sedikit untuk CF, dipakai kesamaan penulis/penerbit."),
    "Rule/Popularitas": ("warning", "**Rule-Based** — user baru (cold-start), ditampilkan buku populer."),
    "Rule/Baseline": ("warning", "**Rule-Based/Baseline** — tidak ada sinyal CF/Content."),
}


# ----------------------------------------------------------------- helpers
def esc(s):
    return re.sub(r"([\\`*_{}\[\]()#+\-.!$~|<>])", r"\\\1", str(s))


def dict_df(d, key, val):
    return pd.DataFrame({key: list(d.keys()), val: list(d.values())})


def bar(d, key, val, color="#4C78A8", height=280):
    df = dict_df(d, key, val)
    ch = (alt.Chart(df).mark_bar(color=color)
          .encode(x=alt.X(f"{key}:N", sort=list(d.keys()), axis=alt.Axis(labelAngle=0)),
                  y=alt.Y(f"{val}:Q"), tooltip=[key, val])
          .properties(height=height))
    st.altair_chart(ch)


def hbar(df, cat, val, color="#4C78A8", height=380):
    ch = (alt.Chart(df).mark_bar(color=color)
          .encode(y=alt.Y(f"{cat}:N", sort="-x", title=None), x=alt.X(f"{val}:Q"),
                  tooltip=list(df.columns))
          .properties(height=height))
    st.altair_chart(ch)


def route_banner(route):
    kind, text = ROUTE_INFO.get(route, ("info", route))
    getattr(st, kind)(f"Route: {text}")


def show_recs(df, key, reason=True):
    """Tampilkan rekomendasi sebagai kartu atau tabel."""
    if df.empty:
        st.warning("Tidak ada rekomendasi yang bisa ditampilkan.")
        return
    view = st.radio("Tampilan", ["Kartu", "Tabel"], horizontal=True, key=f"view_{key}")
    if view == "Kartu":
        per_row = 5
        for s in range(0, len(df), per_row):
            cols = st.columns(per_row)
            for col, (_, r) in zip(cols, df.iloc[s:s + per_row].iterrows()):
                with col:
                    if r.get("Image-URL-M"):
                        st.image(r["Image-URL-M"].replace("http://", "https://"))   # situs https memblokir gambar http
                    st.markdown(f"**{esc(r['Book-Title'])}**")
                    st.caption(f"{r['Author-Display']} · {r['Jml rating']} rating")
                    if reason:
                        st.caption(f"💡 {r['Alasan']}")
    else:
        t = df.drop(columns=[c for c in ["ISBN", "Image-URL-M"] if c in df.columns]).rename(columns={
            "Book-Title": "Judul", "Author-Display": "Penulis", "Year-Of-Publication": "Tahun"})
        if not reason:
            t = t.drop(columns=["Alasan"])
        st.dataframe(t, hide_index=True)


@st.cache_data
def example_users():
    rng = np.random.default_rng(7)
    warm = [u for u, c in rec.user_cnt.items() if c >= rec.min_user_cf and u in rec.cf_users]
    few = [u for u, c in rec.user_cnt.items() if 0 < c < rec.min_user_cf]
    pick = lambda xs: sorted(int(x) for x in rng.choice(xs, min(40, len(xs)), replace=False)) if xs else []
    return {
        f"Warm → CF (≥ {rec.min_user_cf} rating)": pick(warm),
        f"Riwayat sedikit → Content (1–{rec.min_user_cf - 1} rating)": pick(few),
        "User baru (tidak ada di data) → Rule": [999999999],
    }


def pick_user(key):
    mode = st.radio("Pilih user", ["Contoh user", "Input User-ID manual"], horizontal=True, key=f"m_{key}")
    if mode == "Contoh user":
        groups = {g: v for g, v in example_users().items() if v}
        c1, c2 = st.columns(2)
        grp = c1.selectbox("Kelompok user", list(groups), key=f"g_{key}")
        return int(c2.selectbox("User-ID", groups[grp], key=f"u_{key}"))
    return int(st.number_input("User-ID", min_value=0, value=0, step=1, key=f"n_{key}"))


def search_pick(key, only_cf=False, default_q=""):
    q = st.text_input("Cari judul / penulis (kosongkan = buku terpopuler)", value=default_q, key=f"q_{key}")
    opts = rec.search_books(q, 30, only_cf=only_cf)
    if not opts:
        st.warning("Buku tidak ditemukan." + (" (hanya buku yang masuk matriks CF)" if only_cf else ""))
        return None
    return st.selectbox("Pilih buku", opts, format_func=rec.label, key=f"s_{key}")


# ----------------------------------------------------------------- pages
def page_overview():
    st.title("📚 Switching Hybrid Book Recommender")
    st.caption("Dataset Book-Crossing · Item-Based CF + Content-Based + Rule-Based · "
               f"model dibuat {CFG['created_at']}")

    rt = pd.DataFrame(D["rating_eval"]).set_index("Model")
    tk = pd.DataFrame(D["topk_eval"]).set_index("Model/Kelompok")
    c = st.columns(4)
    hyb, base = rt.loc["Switching Hybrid"], rt.loc["Baseline bias (mu+bu+bi)"]
    c[0].metric("RMSE Hybrid", f"{hyb['RMSE']:.3f}", f"{hyb['RMSE'] - base['RMSE']:+.3f} vs baseline",
                delta_color="inverse")
    c[1].metric("MAE Hybrid", f"{hyb['MAE']:.3f}", f"{hyb['MAE'] - base['MAE']:+.3f} vs baseline",
                delta_color="inverse")
    pk = f"Precision@{K}"
    if "Hybrid-warm" in tk.index and "Pop-warm" in tk.index:
        c[2].metric(f"{pk} (user warm)", f"{tk.loc['Hybrid-warm', pk]:.3f}",
                    f"{tk.loc['Hybrid-warm', pk] - tk.loc['Pop-warm', pk]:+.3f} vs populer")
        c[3].metric("Hit-rate (user warm)", f"{tk.loc['Hybrid-warm', 'Hit-rate']:.3f}",
                    f"{tk.loc['Hybrid-warm', 'Hit-rate'] - tk.loc['Pop-warm', 'Hit-rate']:+.3f} vs populer")

    st.subheader("Alur pemrosesan")
    st.markdown("`Preprocessing` → `Split train/test` → `Bias baseline` → `Matriks kemiripan` → "
                "`Prediksi (switching)` → `Evaluasi`")

    st.subheader("Logika switching")
    dot = """
    digraph G { rankdir=LR; bgcolor="transparent";
      node [shape=box, style="rounded,filled", fontname="Helvetica", fillcolor="#EEF3FA", color="#7A8CA5"];
      req [label="Permintaan\\n(user u, buku i)", shape=oval];
      q1  [label="u ≥ %U% rating\\nDAN buku ≥ %I% rating?", shape=diamond, fillcolor="#FFF4D6"];
      cf  [label="1) Item-Based CF\\n(Adjusted Cosine)", fillcolor="#D5F0DD"];
      q2  [label="u punya riwayat?", shape=diamond, fillcolor="#FFF4D6"];
      cb  [label="2) Content-Based\\n(penulis / penerbit)", fillcolor="#D6E8FA"];
      rb  [label="3) Rule-Based\\n(baseline / populer)", fillcolor="#FBE0D2"];
      req -> q1; q1 -> cf [label=" ya"]; q1 -> q2 [label=" tidak"];
      cf -> q2 [label=" tanpa tetangga positif", style=dashed];
      q2 -> cb [label=" ya"]; q2 -> rb [label=" tidak (cold-start)"];
      cb -> rb [label=" tanpa penulis/penerbit cocok", style=dashed];
    }""".replace("%U%", str(rec.min_user_cf)).replace("%I%", str(rec.min_item_cf))
    st.graphviz_chart(dot)

    st.subheader("Porsi tiap route pada data test")
    rs = D["route_share"]
    tot = sum(rs.values())
    st.dataframe(pd.DataFrame({"Route": list(rs), "Jumlah": list(rs.values()),
                               "Porsi (%)": [100 * v / tot for v in rs.values()]}), hide_index=True)
    st.info("Gunakan menu di sidebar: lihat data & EDA, bedah matriks kemiripan, "
            "coba rekomendasi dari berbagai kriteria, dan cek hasil evaluasi.")


def page_data():
    st.title("📂 Data & Preprocessing")
    p = D["pipeline"]
    c = st.columns(4)
    c[0].metric("Rating mentah", f"{p['raw_ratings']:,}")
    c[1].metric("Rating dipakai (1–10)", f"{p['kept_ratings']:,}",
                f"-{p['removed_ratings']:,} dibuang", delta_color="off")
    c[2].metric("User (data mentah)", f"{p['n_users_raw']:,}")
    c[3].metric("Buku (metadata unik)", f"{p['n_books_meta']:,}")

    st.subheader("Tahapan preprocessing")
    st.markdown(
        "1. **Buang rating 0** — rating 0 pada Book-Crossing adalah interaksi implisit, bukan penilaian.\n"
        "2. **Hapus duplikat** pasangan (User-ID, ISBN).\n"
        "3. **Bersihkan metadata** penulis & penerbit (huruf kecil, rapikan `&amp;`) untuk Content-Based.\n"
        "4. **Usia user tidak dipakai model** — banyak kosong dan outlier, hanya untuk eksplorasi.\n"
        "5. **Split acak** 80% train / 20% test (seed tetap).")
    st.dataframe(pd.DataFrame({
        "Tahap": ["Rating mentah", "Eksplisit & unik", "Train", "Test"],
        "Jumlah": [p["raw_ratings"], p["kept_ratings"], p["n_train"], p["n_test"]]}), hide_index=True)

    st.subheader("Cuplikan data")
    t1, t2, t3 = st.tabs(["Ratings", "Books", "Users"])
    t1.dataframe(pd.DataFrame(p["samples"]["ratings"]), hide_index=True)
    t2.dataframe(pd.DataFrame(p["samples"]["books"]), hide_index=True)
    t3.dataframe(pd.DataFrame(p["samples"]["users"]), hide_index=True)

    st.subheader("Kualitas data")
    mv = pd.DataFrame({"Masalah": list(p["missing"]), "Persen": list(p["missing"].values())})
    hbar(mv, "Masalah", "Persen", color="#E45756", height=180)

    st.subheader("Sparsity (kepadatan matriks user × buku)")
    c = st.columns(4)
    c[0].metric("User (train)", f"{p['n_users_train']:,}")
    c[1].metric("Buku (train)", f"{p['n_items_train']:,}")
    c[2].metric("Sparsity penuh", f"{p['sparsity_pct']:.3f}%")
    c[3].metric("Buku di matriks CF", f"{p['n_cf_items']:,}", f"sparsity {p['cf_sparsity_pct']:.2f}%",
                delta_color="off")
    st.caption(f"Hanya buku dengan ≥ {rec.min_item_cf} rating (train) yang masuk matriks kemiripan CF; "
               "sisanya dilayani Content/Rule.")


def page_eda():
    st.title("📊 Exploratory Data Analysis")
    t = st.tabs(["Rating", "User", "Buku", "Penulis & Penerbit", "Demografi"])
    with t[0]:
        st.markdown("**Distribusi rating eksplisit (1–10)**")
        bar(D["rating_dist"], "Rating", "Jumlah")
        st.caption("Rating condong tinggi (8–10) — alasan model memakai bias baseline & residual, bukan rating mentah.")
        st.markdown("**Rata-rata rating per user**")
        bar(D["user_mean_hist"], "Rentang rata-rata", "Jumlah user", color="#F58518")
    with t[1]:
        st.markdown("**Jumlah rating per user** — long tail: mayoritas user hanya menilai sedikit buku")
        bar(D["user_activity_hist"], "Jumlah rating", "Jumlah user", color="#54A24B")
    with t[2]:
        st.markdown("**Jumlah rating per buku**")
        bar(D["item_activity_hist"], "Jumlah rating", "Jumlah buku", color="#B279A2")
        st.markdown("**20 buku paling banyak dinilai**")
        st.dataframe(pd.DataFrame(D["top_books"]).drop(columns="ISBN").rename(columns={
            "title": "Judul", "author": "Penulis", "n_ratings": "Jml rating", "mean": "Rata-rata"}),
            hide_index=True)
    with t[3]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Penulis teratas (jumlah rating)**")
            hbar(pd.DataFrame(D["top_authors"]).rename(columns={"Book-Author": "Penulis", "n": "Jumlah rating"}),
                 "Penulis", "Jumlah rating", height=420)
        with c2:
            st.markdown("**Penerbit teratas (jumlah rating)**")
            hbar(pd.DataFrame(D["top_publishers"]).rename(columns={"Publisher": "Penerbit", "n": "Jumlah rating"}),
                 "Penerbit", "Jumlah rating", color="#F58518", height=420)
    with t[4]:
        st.markdown("**Distribusi usia user** (5–100 tahun; hanya eksplorasi, tidak dipakai model)")
        bar(D["age_hist"], "Rentang usia", "Jumlah user", color="#72B7B2")
        if D["year_hist"]:
            st.markdown("**Tahun terbit buku**")
            bar(D["year_hist"], "Rentang tahun", "Jumlah buku", color="#9D755D")


def page_model():
    st.title("🧮 Model & Matriks Kemiripan")
    st.subheader("Hyperparameter")
    names = {"MIN_ITEM_CF": "Min. rating per buku (masuk matriks CF)", "MIN_USER_CF": "Min. rating per user (dilayani CF)",
             "K_NEIGHBORS": "Jumlah tetangga terdekat", "SHRINK": "Shrinkage kemiripan",
             "REG_BIAS": "Regularisasi bias", "DAMP": "Damping penyebut prediksi",
             "TOP_K": "K untuk Top-K", "REL_THRESHOLD": "Ambang rating relevan"}
    st.dataframe(pd.DataFrame({"Parameter": list(names), "Keterangan": list(names.values()),
                               "Nilai": [CFG[k] for k in names]}), hide_index=True)

    st.subheader("Statistik matriks kemiripan")
    S = rec.S
    n = S.shape[0]
    vals = S.data
    c = st.columns(4)
    c[0].metric("Ukuran matriks", f"{n:,} × {n:,}")
    c[1].metric("Pasangan sim > 0", f"{S.nnz:,}")
    c[2].metric("Kepadatan", f"{100 * S.nnz / (n * n):.2f}%")
    c[3].metric("Rata-rata sim positif", f"{vals.mean():.3f}" if len(vals) else "-")
    if len(vals):
        cnt, edges = np.histogram(vals, bins=20, range=(0, max(float(vals.max()), 1e-6)))
        hist = pd.DataFrame({"Kemiripan": [f"{a:.2f}" for a in edges[:-1]], "Jumlah pasangan": cnt})
        st.markdown("**Distribusi nilai kemiripan positif**")
        st.altair_chart(alt.Chart(hist).mark_bar(color="#4C78A8")
                        .encode(x=alt.X("Kemiripan:N", sort=list(hist["Kemiripan"])), y="Jumlah pasangan:Q",
                                tooltip=["Kemiripan", "Jumlah pasangan"]).properties(height=260))

    st.subheader("Heatmap kemiripan antar-buku terpopuler")
    top_n = st.slider("Jumlah buku", 10, min(40, n), min(20, n))
    top = sorted(rec.cf_items, key=lambda i: -rec.item_cnt[i])[:top_n]
    ix = [rec.item_ix[i] for i in top]
    sub = S[ix][:, ix].toarray()
    labels = [f"{k + 1}. {rec.title(i)[:26]}" for k, i in enumerate(top)]
    hm = pd.DataFrame({"x": np.tile(labels, top_n), "y": np.repeat(labels, top_n), "Kemiripan": sub.ravel()})
    st.altair_chart(alt.Chart(hm).mark_rect()
                    .encode(x=alt.X("x:N", sort=labels, title=None, axis=alt.Axis(labelLimit=160)),
                            y=alt.Y("y:N", sort=labels, title=None, axis=alt.Axis(labelLimit=160)),
                            color=alt.Color("Kemiripan:Q", scale=alt.Scale(scheme="blues")),
                            tooltip=["x", "y", alt.Tooltip("Kemiripan:Q", format=".3f")])
                    .properties(height=max(320, 22 * top_n)))
    st.caption("Adjusted cosine pada rating yang dipusatkan per user, dengan shrinkage agar pasangan "
               "dengan sedikit irisan user tidak mendapat kemiripan palsu. Warna kosong = 0.")

    st.subheader("Cari buku mirip")
    isbn = search_pick("sim", only_cf=True)
    if isbn:
        kk = st.slider("Jumlah buku mirip", 5, 20, 10, key="sim_k")
        df = rec.similar_items(isbn, kk)
        t = df[["Book-Title", "Author-Display", "Skor", "Jml rating"]].rename(columns={
            "Book-Title": "Judul", "Author-Display": "Penulis", "Skor": "Kemiripan"})
        st.dataframe(t, hide_index=True, column_config={
            "Kemiripan": st.column_config.ProgressColumn("Kemiripan", min_value=0.0, max_value=1.0, format="%.3f")})


def page_reco():
    st.title("🎯 Coba Rekomendasi")
    st.caption("Satu model, beberapa cara masuk. Coba dari sudut pandang yang berbeda.")
    t = st.tabs(["👤 Berdasarkan User", "📖 Berdasarkan Buku Favorit", "🔎 Populer & Filter", "🆕 Profil Baru (Cold-Start)"])

    with t[0]:
        uid = pick_user("reco")
        k = st.slider("Jumlah rekomendasi", 5, 20, K, key="k_user")
        n = rec.user_cnt.get(uid, 0)
        c = st.columns(3)
        c[0].metric("User-ID", uid)
        c[1].metric("Jumlah rating (train)", n)
        hist = rec.history[rec.history["User-ID"] == uid]
        c[2].metric("Rata-rata rating", f"{hist['Book-Rating'].mean():.2f}" if len(hist) else "-")
        df, route = rec.recommend(uid, k)
        route_banner(route)
        if n:
            with st.expander("Buku yang dinilai tinggi oleh user ini"):
                h = rec.user_history(uid, 8)[["Book-Rating", "Book-Title", "Author-Display"]]
                st.dataframe(h.rename(columns={"Book-Rating": "Rating", "Book-Title": "Judul",
                                               "Author-Display": "Penulis"}), hide_index=True)
        show_recs(df, "user")

    with t[1]:
        isbn = search_pick("fav")
        if isbn:
            k = st.slider("Jumlah rekomendasi", 5, 20, K, key="k_fav")
            st.markdown(f"Buku acuan: **{esc(rec.title(isbn))}** — {rec.author(isbn)}")
            if isbn in rec.item_ix:
                st.success("Route: **Item-Based CF** — diambil dari baris matriks kemiripan buku ini.")
                show_recs(rec.similar_items(isbn, k), "fav")
            else:
                st.warning(f"Buku ini punya < {rec.min_item_cf} rating sehingga tidak ada di matriks CF. "
                           "Fallback **Content-Based**: buku lain dari penulis yang sama.")
                show_recs(rec.popular(k, author=rec.author(isbn), min_ratings=1), "fav2")

    with t[2]:
        c1, c2 = st.columns(2)
        auth = c1.selectbox("Penulis", ["(semua)"] + rec.top_values("Author-Display"))
        pub = c2.selectbox("Penerbit", ["(semua)"] + rec.top_values("Publisher"),
                           format_func=lambda x: x if x == "(semua)" else x.title())
        c3, c4, c5 = st.columns(3)
        yr = pd.to_numeric(rec.books.get("Year-Of-Publication", pd.Series(dtype=str)), errors="coerce")
        yr = yr[(yr >= 1900) & (yr <= 2010)]
        rng = c3.slider("Tahun terbit", 1900, 2010, (1900, 2010)) if len(yr) else None
        minr = c4.slider("Min. jumlah rating", 1, 50, 5)
        k = c5.slider("Jumlah buku", 5, 20, K, key="k_pop")
        st.caption("Rule-Based: urut berdasarkan rata-rata rating ter-damping (buku dengan sedikit rating "
                   "ditarik ke rata-rata global).")
        show_recs(rec.popular(k, None if auth == "(semua)" else auth, None if pub == "(semua)" else pub,
                              rng, minr), "pop", reason=False)

    with t[3]:
        st.markdown("Bayangkan kamu pengguna baru: beri rating beberapa buku, model langsung membuat profil "
                    "dan merekomendasikan buku lain (CF dari rating yang kamu isi).")
        prof = st.session_state.setdefault("profile", {})
        isbn = search_pick("prof", only_cf=True)
        c1, c2, c3 = st.columns([2, 1, 1])
        r = c1.slider("Rating untuk buku ini", 1, 10, 8, key="prof_r")
        if c2.button("➕ Tambah ke profil", disabled=isbn is None):
            prof[isbn] = r
            st.rerun()
        if c3.button("🎲 Isi contoh acak"):
            pool = sorted(rec.cf_items, key=lambda i: -rec.item_cnt[i])[:100]
            rg = np.random.default_rng()
            for i in rg.choice(pool, min(5, len(pool)), replace=False):
                prof[str(i)] = int(rg.integers(7, 11))
            st.rerun()
        if prof:
            st.markdown("**Profil kamu**")
            st.dataframe(pd.DataFrame({"Judul": [rec.title(i) for i in prof], "Penulis": [rec.author(i) for i in prof],
                                       "Rating": list(prof.values())}), hide_index=True)
            if st.button("🗑️ Kosongkan profil"):
                prof.clear()
                st.rerun()
            k = st.slider("Jumlah rekomendasi", 5, 20, K, key="k_prof")
            df, route = rec.recommend_profile(prof, k)
            route_banner(route)
            show_recs(df, "prof")
        else:
            st.info("Profil masih kosong — tambahkan minimal 1 buku, atau klik 'Isi contoh acak'.")


def page_predict():
    st.title("⭐ Prediksi Rating")
    st.caption("Berapa rating yang akan diberikan user ke sebuah buku? Lihat juga bagaimana angkanya terbentuk.")
    uid = pick_user("pred")
    isbn = search_pick("pred")
    if not isbn:
        return
    d = rec.predict_detail(uid, isbn)
    h = rec.history[(rec.history["User-ID"] == uid) & (rec.history["ISBN"] == isbn)]
    c = st.columns(4)
    c[0].metric("Prediksi rating", f"{d['pred']:.2f}")
    c[1].metric("Baseline (μ+bᵤ+bᵢ)", f"{d['baseline']:.2f}")
    c[2].metric("Route", d["route"])
    c[3].metric("Rating asli (train)", int(h['Book-Rating'].iloc[0]) if len(h) else "belum menilai")
    route_banner(d["route"])
    st.caption(d["note"])

    st.subheader("Komponen baseline")
    st.dataframe(pd.DataFrame({
        "Komponen": ["μ (rata-rata global)", "bᵤ (bias user)", "bᵢ (bias buku)"],
        "Nilai": [d["mu"], d["b_u"], d["b_i"]]}), hide_index=True)
    st.caption("bᵤ = 0 untuk user baru dan bᵢ = 0 untuk buku baru.")
    if d["neighbors"] is not None:
        st.subheader("Buku mirip yang dinilai user (tetangga CF)")
        st.dataframe(d["neighbors"].drop(columns="ISBN"), hide_index=True, column_config={
            "Kemiripan": st.column_config.ProgressColumn("Kemiripan", min_value=0.0, max_value=1.0, format="%.3f")})
        st.caption("Prediksi = baseline + Σ(kemiripan × residual) / (Σ|kemiripan| + damping).")


def page_eval():
    st.title("📈 Evaluasi Model")
    st.subheader("Prediksi rating (data test)")
    rt = pd.DataFrame(D["rating_eval"])
    c1, c2 = st.columns([1, 1])
    c1.dataframe(rt.round(4), hide_index=True)
    long = rt.melt("Model", var_name="Metrik", value_name="Nilai")
    c2.altair_chart(alt.Chart(long).mark_bar().encode(
        x=alt.X("Model:N", title=None, axis=alt.Axis(labelAngle=-15, labelLimit=140)), y="Nilai:Q",
        color="Metrik:N", xOffset="Metrik:N", tooltip=["Model", "Metrik", alt.Tooltip("Nilai:Q", format=".4f")])
        .properties(height=260))
    st.caption("Semakin kecil semakin baik. Hybrid dibandingkan dengan rata-rata global dan baseline bias.")

    st.subheader("Rincian per route")
    ro = pd.DataFrame(D["route_eval"])
    st.dataframe(ro.round(4), hide_index=True)
    l = ro.melt("Route", ["RMSE hybrid", "RMSE baseline"], "Model", "RMSE")
    st.altair_chart(alt.Chart(l).mark_bar().encode(
        x=alt.X("Route:N", axis=alt.Axis(labelAngle=0)), y="RMSE:Q", color="Model:N", xOffset="Model:N",
        tooltip=["Route", "Model", alt.Tooltip("RMSE:Q", format=".4f")]).properties(height=260))

    st.subheader(f"Top-{K}: Precision, Recall, Hit-rate")
    tk = pd.DataFrame(D["topk_eval"])
    if len(tk):
        st.dataframe(tk.round(4), hide_index=True)
        l = tk.melt("Model/Kelompok", [f"Precision@{K}", f"Recall@{K}", "Hit-rate"], "Metrik", "Nilai")
        st.altair_chart(alt.Chart(l).mark_bar().encode(
            x=alt.X("Model/Kelompok:N", axis=alt.Axis(labelAngle=0), title=None), y="Nilai:Q",
            color="Metrik:N", xOffset="Metrik:N",
            tooltip=["Model/Kelompok", "Metrik", alt.Tooltip("Nilai:Q", format=".4f")]).properties(height=280))
        st.caption(f"Relevan = rating test ≥ {CFG['REL_THRESHOLD']}. 'warm' = user dengan ≥ {rec.min_user_cf} rating; "
                   "'cold' = riwayat sedikit. 'Pop' = rekomendasi populer sebagai pembanding.")

    st.subheader("Prediksi vs aktual")
    ts = load_test_sample()
    routes = st.multiselect("Route", sorted(ts["route"].unique()), default=sorted(ts["route"].unique()))
    ts = ts[ts["route"].isin(routes)].copy()
    if len(ts):
        ts["Rating aktual (jitter)"] = ts["Book-Rating"] + np.random.default_rng(0).normal(0, 0.12, len(ts))
        ts["Galat"] = ts["pred"] - ts["Book-Rating"]
        pts = alt.Chart(ts).mark_circle(size=18, opacity=0.25).encode(
            x=alt.X("Rating aktual (jitter):Q", scale=alt.Scale(domain=[0.5, 10.5])),
            y=alt.Y("pred:Q", title="Prediksi", scale=alt.Scale(domain=[0.5, 10.5])), color="route:N")
        diag = alt.Chart(pd.DataFrame({"a": [1, 10], "b": [1, 10]})).mark_line(color="gray", strokeDash=[4, 4]) \
            .encode(x="a:Q", y="b:Q")
        st.altair_chart((pts + diag).properties(height=360))
        st.markdown("**Distribusi galat (prediksi − aktual)**")
        st.altair_chart(alt.Chart(ts).mark_bar(opacity=0.7).encode(
            x=alt.X("Galat:Q", bin=alt.Bin(maxbins=40)), y="count():Q", color="route:N").properties(height=240))
        st.markdown("**Rata-rata prediksi per rating aktual** (idealnya mengikuti garis diagonal)")
        cal = ts.groupby("Book-Rating")["pred"].mean().reset_index().rename(
            columns={"Book-Rating": "Rating aktual", "pred": "Rata-rata prediksi"})
        st.altair_chart(alt.Chart(cal).mark_line(point=True).encode(
            x="Rating aktual:Q", y=alt.Y("Rata-rata prediksi:Q", scale=alt.Scale(domain=[1, 10]))).properties(height=240))


PAGES = {
    "🏠 Ringkasan & Alur": page_overview,
    "📂 Data & Preprocessing": page_data,
    "📊 EDA": page_eda,
    "🧮 Model & Matriks Kemiripan": page_model,
    "🎯 Coba Rekomendasi": page_reco,
    "⭐ Prediksi Rating": page_predict,
    "📈 Evaluasi": page_eval,
}
st.sidebar.title("📚 Book Recommender")
choice = st.sidebar.radio("Menu", list(PAGES), key="nav")
st.sidebar.caption(f"{CFG['n_books']:,} buku · {CFG['n_users']:,} user · {CFG['n_train']:,} rating (train)")
PAGES[choice]()
