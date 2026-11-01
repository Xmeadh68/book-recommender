"""
Switching Hybrid Recommender - Book-Crossing
=============================================
Alur : Preprocessing -> Split train/test -> Build model -> Prediksi rating -> Evaluasi
Route (switching) :
  1) Item-Based CF (Adjusted Cosine)  : user punya >= MIN_USER_CF rating & buku punya >= MIN_ITEM_CF rating
  2) Content-Based (penulis/penerbit) : user punya riwayat tetapi syarat CF tidak terpenuhi
  3) Rule-Based (popularitas/baseline): user/buku benar-benar baru (cold-start)

Cara pakai:  python book_recommender.py   (ubah DATA_DIR di bawah bila perlu)
Butuh     :  pandas, numpy, scipy
"""
import time
import numpy as np
import pandas as pd
from scipy import sparse

# ----------------------------- KONFIGURASI -----------------------------
import os
import glob
import shutil
ON_KAGGLE = os.path.isdir("/kaggle/working")
DATA_DIR = "data"          # lokal: folder yang berisi BX-*.csv
OUT_DIR = "."              # lokal: tempat CSV hasil & folder artifacts/
if ON_KAGGLE:              # Kaggle: dataset dicari otomatis, output ke /kaggle/working
    _hits = glob.glob("/kaggle/input/**/BX-Book-Ratings.csv", recursive=True)
    if not _hits:
        raise FileNotFoundError("BX-Book-Ratings.csv tidak ditemukan di /kaggle/input. "
                                "Pastikan dataset Book-Crossing sudah di-Add Data pada notebook.")
    DATA_DIR = os.path.dirname(_hits[0])
    OUT_DIR = "/kaggle/working"
SEED = 42
TEST_SIZE = 0.20
MIN_ITEM_CF = 10      # buku minimal punya n rating (train) agar masuk matriks kemiripan
MIN_USER_CF = 3       # user minimal punya n rating (train) agar dilayani CF
K_NEIGHBORS = 30      # jumlah tetangga terdekat pada prediksi rating
SHRINK = 10           # shrinkage kemiripan (menahan sim palsu dari sedikit irisan)
REG_BIAS = 10         # regularisasi bias user/item (damping)
DAMP = 1.0            # damping penyebut prediksi CF: sum|sim| + DAMP (menahan koreksi dari tetangga lemah)
TOP_K = 10            # K untuk Precision@K / Recall@K
REL_THRESHOLD = 8     # rating test >= 8 dianggap "relevan"
MAX_EVAL_USERS = 3000
rng = np.random.default_rng(SEED)
t0 = time.time()


def log(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)


# =========================== 1. DATA PREPROCESSING ===========================
log("Memuat data ...")
kw = dict(sep=";", encoding="latin-1", dtype=str, on_bad_lines="skip")
ratings = pd.read_csv(f"{DATA_DIR}/BX-Book-Ratings.csv", **kw)
books = pd.read_csv(f"{DATA_DIR}/BX-Books.csv", **kw)
users = pd.read_csv(f"{DATA_DIR}/BX-Users.csv", **kw)

ratings["Book-Rating"] = ratings["Book-Rating"].astype(int)
ratings["User-ID"] = ratings["User-ID"].astype(int)
log(f"Raw ratings={len(ratings):,}  books={len(books):,}  users={len(users):,}")

# a) Rating 0 = implisit (bukan penilaian) -> dibuang untuk pemodelan rating
n_all = len(ratings)
ratings = ratings[ratings["Book-Rating"] > 0].copy()
log(f"Rating eksplisit (1-10): {len(ratings):,} dari {n_all:,}")

# b) Duplikat (user, ISBN)
ratings = ratings.drop_duplicates(["User-ID", "ISBN"])

# c) Metadata buku: bersihkan penulis & penerbit untuk Content-Based
books = books.drop_duplicates("ISBN").copy()
for c in ["Book-Author", "Publisher"]:
    books[c] = (books[c].fillna("").str.lower().str.strip()
                .str.replace("&amp;", "&", regex=False))
meta = books.set_index("ISBN")[["Book-Title", "Book-Author", "Publisher"]]
ratings = ratings.join(meta[["Book-Author", "Publisher"]], on="ISBN")
ratings[["Book-Author", "Publisher"]] = ratings[["Book-Author", "Publisher"]].fillna("")

# d) Usia user hanya untuk eksplorasi (39,7% kosong, banyak outlier) -> tidak dipakai model
users["Age"] = pd.to_numeric(users["Age"], errors="coerce")

# =========================== 2. SPLIT TRAIN / TEST ===========================
idx = rng.permutation(len(ratings))
n_test = int(len(ratings) * TEST_SIZE)
test = ratings.iloc[idx[:n_test]].reset_index(drop=True)
train = ratings.iloc[idx[n_test:]].reset_index(drop=True)
log(f"Train={len(train):,}  Test={len(test):,}")

# =========================== 3. BUILD MODEL ===========================
# --- 3a. Baseline bias (Rule-Based / fallback): mu + b_u + b_i ---
mu = train["Book-Rating"].mean()
item_g = train.groupby("ISBN")["Book-Rating"].agg(["sum", "count"])
b_i = ((item_g["sum"] - mu * item_g["count"]) / (item_g["count"] + REG_BIAS)).to_dict()
train["_bi"] = train["ISBN"].map(b_i).fillna(0.0)
user_g = (train["Book-Rating"] - mu - train["_bi"]).groupby(train["User-ID"]).agg(["sum", "count"])
b_u = (user_g["sum"] / (user_g["count"] + REG_BIAS)).to_dict()
train["_bu"] = train["User-ID"].map(b_u).fillna(0.0)
train["_base"] = mu + train["_bi"] + train["_bu"]
train["_res"] = train["Book-Rating"] - train["_base"]          # residual terhadap baseline
user_cnt = train.groupby("User-ID").size().to_dict()
item_cnt = train.groupby("ISBN").size().to_dict()


def baseline(u, i):
    return mu + b_i.get(i, 0.0) + b_u.get(u, 0.0)


# --- 3b. Item-Based CF: Adjusted Cosine pada residual ---
cf_items = sorted([i for i, c in item_cnt.items() if c >= MIN_ITEM_CF])
item_ix = {i: k for k, i in enumerate(cf_items)}
n_items = len(cf_items)
log(f"Buku masuk matriks kemiripan CF: {n_items:,}")

cf_tr = train[train["ISBN"].isin(item_ix)]
u_ids = sorted(train["User-ID"].unique())
user_ix = {u: k for k, u in enumerate(u_ids)}
rows = cf_tr["User-ID"].map(user_ix).values
cols = cf_tr["ISBN"].map(item_ix).values
# Adjusted cosine: pusatkan rating per user (mean user) lalu normalkan kolom
umean = train.groupby("User-ID")["Book-Rating"].mean()
vals = (cf_tr["Book-Rating"] - cf_tr["User-ID"].map(umean)).values
X = sparse.csr_matrix((vals, (rows, cols)), shape=(len(u_ids), n_items))
Bin = sparse.csr_matrix((np.ones(len(vals)), (rows, cols)), shape=X.shape)

num = (X.T @ X).toarray().astype(np.float32)
norm = np.sqrt(np.asarray(X.multiply(X).sum(axis=0)).ravel()).astype(np.float32)
den = np.outer(norm, norm)
S = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
co = (Bin.T @ Bin).toarray().astype(np.float32)               # jumlah user yang menilai kedua buku
S *= co / (co + SHRINK)                                         # shrinkage
np.fill_diagonal(S, 0.0)
del num, den, co
log(f"Matriks kemiripan {S.shape} selesai")

# --- 3c. Content-Based: rata-rata residual user per penulis / penerbit ---
auth_res = (train[train["Book-Author"] != ""]
            .groupby(["User-ID", "Book-Author"])["_res"].agg(["sum", "count"]))
pub_res = (train[train["Publisher"] != ""]
           .groupby(["User-ID", "Publisher"])["_res"].agg(["sum", "count"]))
auth_d = {k: (s, c) for k, s, c in zip(auth_res.index, auth_res["sum"], auth_res["count"])}
pub_d = {k: (s, c) for k, s, c in zip(pub_res.index, pub_res["sum"], pub_res["count"])}

# riwayat per user (untuk CF)
user_hist = {}
for u, g in cf_tr.groupby("User-ID"):
    user_hist[u] = (g["ISBN"].map(item_ix).values, g["_res"].values)

# =========================== 4. PREDIKSI RATING (SWITCHING) ===========================
def predict_cf(u, i):
    """Item-Based CF -> None bila tidak ada tetangga positif."""
    if i not in item_ix or u not in user_hist:
        return None
    js, res = user_hist[u]
    s = S[item_ix[i], js]
    pos = s > 0
    if not pos.any():
        return None
    s, res = s[pos], res[pos]
    if len(s) > K_NEIGHBORS:
        top = np.argpartition(-s, K_NEIGHBORS)[:K_NEIGHBORS]
        s, res = s[top], res[top]
    return baseline(u, i) + float(np.dot(s, res) / (np.abs(s).sum() + DAMP))


def predict_cb(u, i, author, pub):
    for d, key in ((auth_d, (u, author)), (pub_d, (u, pub))):
        if key[1] and key in d:
            s, c = d[key]
            return baseline(u, i) + (s / c) * (c / (c + 2.0))
    return None


def predict_hybrid(u, i, author, pub):
    """Return (prediksi, route)."""
    if user_cnt.get(u, 0) >= MIN_USER_CF and item_cnt.get(i, 0) >= MIN_ITEM_CF:
        p = predict_cf(u, i)
        if p is not None:
            return float(np.clip(p, 1, 10)), "CF"
    if user_cnt.get(u, 0) > 0:
        p = predict_cb(u, i, author, pub)
        if p is not None:
            return float(np.clip(p, 1, 10)), "Content"
    return float(np.clip(baseline(u, i), 1, 10)), "Rule/Baseline"


log("Memprediksi rating data test ...")
preds, routes, base_preds = [], [], []
for u, i, a, p in zip(test["User-ID"], test["ISBN"], test["Book-Author"], test["Publisher"]):
    pr, rt = predict_hybrid(u, i, a, p)
    preds.append(pr)
    routes.append(rt)
    base_preds.append(float(np.clip(baseline(u, i), 1, 10)))
test["pred"], test["route"], test["base"] = preds, routes, base_preds
test["global_mean"] = mu


def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))
def mae(y, p): return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


# =========================== 5. EVALUASI RATING (RMSE / MAE) ===========================
print("\n" + "=" * 70)
print("EVALUASI PREDIKSI RATING (data test)")
print("=" * 70)
y = test["Book-Rating"]
res_tbl = pd.DataFrame({
    "Model": ["Rata-rata global", "Baseline bias (mu+bu+bi)", "Switching Hybrid"],
    "RMSE": [rmse(y, test["global_mean"]), rmse(y, test["base"]), rmse(y, test["pred"])],
    "MAE": [mae(y, test["global_mean"]), mae(y, test["base"]), mae(y, test["pred"])],
})
print(res_tbl.round(4).to_string(index=False))

print("\nRincian per route (switching):")
rows_ = []
for rt, g in test.groupby("route"):
    rows_.append({"Route": rt, "N test": len(g), "Porsi": len(g) / len(test),
                  "RMSE hybrid": rmse(g["Book-Rating"], g["pred"]),
                  "RMSE baseline": rmse(g["Book-Rating"], g["base"]),
                  "MAE hybrid": mae(g["Book-Rating"], g["pred"]),
                  "MAE baseline": mae(g["Book-Rating"], g["base"])})
route_tbl = pd.DataFrame(rows_)
print(route_tbl.round(4).to_string(index=False))

# =========================== 6. EVALUASI TOP-K (Precision@K / Recall@K) ===========================
pool_pop = [i for i, c in item_cnt.items() if c >= 3]
pop_score = {i: (item_g.loc[i, "sum"] + mu * 5) / (item_g.loc[i, "count"] + 5) for i in pool_pop}
# popularitas: rata-rata ter-damping, tie-break jumlah rating -> urutan tetap
pop_rank = sorted(pool_pop, key=lambda i: (-pop_score[i], -item_cnt[i]))[:2000]
pop_rank_ix = [i for i in pop_rank]
pop_cf_rank = [i for i in pop_rank if i in item_ix]

author_items = (train[train["Book-Author"] != ""].groupby("Book-Author")["ISBN"]
                .apply(lambda s: s.value_counts().index.tolist()).to_dict())

test_rel = (test[test["Book-Rating"] >= REL_THRESHOLD].groupby("User-ID")["ISBN"].apply(set).to_dict())
train_seen = train.groupby("User-ID")["ISBN"].apply(set).to_dict()
elig = [u for u in test_rel if u in train_seen]
elig = list(rng.permutation(elig))[:MAX_EVAL_USERS]
log(f"User evaluasi Top-{TOP_K}: {len(elig):,}")

item_arr = np.array(cf_items)


def rec_cf(u):
    js, _ = user_hist[u]
    w = user_w[u]
    sc = np.maximum(S[:, js], 0) @ w
    sc[js] = -1                                        # buang yang sudah dibaca
    top = np.argpartition(-sc, TOP_K + 5)[:TOP_K + 5]
    top = top[np.argsort(-sc[top])]
    seen = train_seen[u]
    return [item_arr[t] for t in top if item_arr[t] not in seen and sc[t] > 0][:TOP_K]


def rec_content(u):
    liked = train[(train["User-ID"] == u) & (train["Book-Rating"] >= REL_THRESHOLD)]
    seen, out = train_seen[u], []
    for a in liked["Book-Author"]:
        for i in author_items.get(a, [])[:5]:
            if i not in seen and i not in out:
                out.append(i)
    return out[:TOP_K]


def rec_pop(u):
    seen = train_seen[u]
    return [i for i in pop_rank if i not in seen][:TOP_K]


# bobot user untuk CF: rating (skala 0-1) pada buku di matriks CF
user_w = {u: (cf_tr[cf_tr["User-ID"] == u]["Book-Rating"].values / 10.0)
          for u in elig if u in user_hist}

metrics = {k: {"P": [], "R": [], "H": []} for k in ["Hybrid-warm", "Pop-warm", "Hybrid-cold", "Pop-cold"]}


def score(recs, rel):
    hit = len(set(recs) & rel)
    return hit / TOP_K, hit / len(rel), float(hit > 0)


for u in elig:
    rel = test_rel[u]
    warm = user_cnt.get(u, 0) >= MIN_USER_CF and u in user_hist
    if warm:
        recs = rec_cf(u)
        if len(recs) < TOP_K:                              # lengkapi dengan popularitas
            recs += [i for i in rec_pop(u) if i not in recs][:TOP_K - len(recs)]
        grp = "warm"
    else:
        recs = rec_content(u)
        if len(recs) < TOP_K:
            recs += [i for i in rec_pop(u) if i not in recs][:TOP_K - len(recs)]
        grp = "cold"
    for name, r in ((f"Hybrid-{grp}", recs), (f"Pop-{grp}", rec_pop(u))):
        p, rc, h = score(r, rel)
        metrics[name]["P"].append(p); metrics[name]["R"].append(rc); metrics[name]["H"].append(h)

print("\n" + "=" * 70)
print(f"EVALUASI TOP-{TOP_K} (relevan = rating test >= {REL_THRESHOLD})")
print("=" * 70)
rows_ = []
for k, m in metrics.items():
    if m["P"]:
        rows_.append({"Model/Kelompok": k, "N user": len(m["P"]),
                      f"Precision@{TOP_K}": np.mean(m["P"]),
                      f"Recall@{TOP_K}": np.mean(m["R"]),
                      "Hit-rate": np.mean(m["H"])})
topk_tbl = pd.DataFrame(rows_)
print(topk_tbl.round(4).to_string(index=False))

# =========================== 7. CONTOH REKOMENDASI ===========================
print("\n" + "=" * 70)
print("CONTOH REKOMENDASI (1 user warm)")
print("=" * 70)
demo = next((u for u in elig if user_cnt.get(u, 0) >= 15 and u in user_hist), None)
if demo is not None:
    liked = train[(train["User-ID"] == demo)].sort_values("Book-Rating", ascending=False).head(5)
    print(f"User {demo} - buku yang dinilai tinggi:")
    for i, r in zip(liked["ISBN"], liked["Book-Rating"]):
        print(f"  [{r}] {meta['Book-Title'].get(i, i)}")
    print("Rekomendasi:")
    for i in rec_cf(demo)[:TOP_K]:
        print(f"  - {meta['Book-Title'].get(i, i)}  ({meta['Book-Author'].get(i, '')})")

# simpan hasil
res_tbl.to_csv(os.path.join(OUT_DIR, "hasil_rating_rmse_mae.csv"), index=False)
route_tbl.to_csv(os.path.join(OUT_DIR, "hasil_per_route.csv"), index=False)
topk_tbl.to_csv(os.path.join(OUT_DIR, "hasil_topk.csv"), index=False)
log("Selesai. Tabel hasil disimpan ke CSV.")


# =========================== 8. SIMPAN ARTEFAK UNTUK STREAMLIT ===========================
import os
import json
import pickle
import datetime

ART_DIR = os.path.join(OUT_DIR, "artifacts")
PRUNE_TOPN = None   # None = eksak (hasil identik). Isi mis. 100 untuk memperkecil ukuran file S
os.makedirs(ART_DIR, exist_ok=True)


def _size(path):
    return f"{os.path.getsize(path) / 1e6:.2f} MB"


def _records(df):
    """DataFrame -> list of dict yang aman untuk JSON (NaN -> null)."""
    return json.loads(df.to_json(orient="records", force_ascii=False))


def _hist(series, bins, labels=None):
    h = pd.cut(series, bins=bins, labels=labels).value_counts().sort_index()
    return {str(k): int(v) for k, v in h.items()}


# --- 8a. Matriks kemiripan (hanya nilai positif -> eksak, karena prediksi & rekomendasi
#         memang hanya memakai sim > 0). Disimpan sparse + terkompresi. ---
S_art = np.maximum(S, 0).astype(np.float32)
if PRUNE_TOPN and PRUNE_TOPN < n_items:
    drop = np.argpartition(-S_art, PRUNE_TOPN, axis=1)[:, PRUNE_TOPN:]
    np.put_along_axis(S_art, drop, 0.0, axis=1)
S_sp = sparse.csr_matrix(S_art)
sparse.save_npz(f"{ART_DIR}/similarity.npz", S_sp, compressed=True)
log(f"similarity.npz  nnz={S_sp.nnz:,}  ({_size(ART_DIR + '/similarity.npz')})")
del S_art

# --- 8b. State model (sengaja kecil) ---
# Bias user/buku, riwayat CF, tabel penulis/penerbit & popularitas TIDAK disimpan di sini:
# semuanya dihitung ulang saat dashboard dimuat dari user_history.csv.gz (lihat recommender_core.py).
# Yang wajib disimpan hanya urutan buku pada matriks kemiripan.
model_state = {"mu": float(mu), "cf_items": list(cf_items)}
with open(f"{ART_DIR}/model_state.pkl", "wb") as f:
    pickle.dump(model_state, f, protocol=pickle.HIGHEST_PROTOCOL)
log(f"model_state.pkl ({_size(ART_DIR + '/model_state.pkl')})")

# --- 8c. Metadata buku (hanya yang muncul di train) + riwayat rating user ---
img_cols = [c for c in ["Image-URL-M"] if c in books.columns]   # dashboard hanya memakai ukuran M
book_cols = ["ISBN", "Book-Title", "Book-Author", "Publisher"]
if "Year-Of-Publication" in books.columns:
    book_cols.append("Year-Of-Publication")
books_art = books[books["ISBN"].isin(item_cnt)][book_cols + img_cols].copy()
books_art["Author-Display"] = books_art["Book-Author"].str.title()   # tampilan (kolom asli huruf kecil)
books_art.to_csv(f"{ART_DIR}/books.csv.gz", index=False)
train[["User-ID", "ISBN", "Book-Rating"]].to_csv(f"{ART_DIR}/user_history.csv.gz", index=False)
log(f"books.csv.gz ({_size(ART_DIR + '/books.csv.gz')}), "
    f"user_history.csv.gz ({_size(ART_DIR + '/user_history.csv.gz')})")

# --- 8d. Data ringkasan dashboard: preprocessing, EDA, evaluasi ---
n_users_tr, n_items_tr = train["User-ID"].nunique(), train["ISBN"].nunique()
pipeline = {
    "raw_ratings": int(n_all),
    "kept_ratings": int(len(ratings)),                 # eksplisit (1-10) & tanpa duplikat
    "removed_ratings": int(n_all - len(ratings)),      # rating 0 (implisit) + duplikat
    "n_users_raw": int(len(users)),
    "n_books_meta": int(len(books)),                   # buku unik di metadata
    "n_train": int(len(train)), "n_test": int(len(test)),
    "n_users_train": int(n_users_tr), "n_items_train": int(n_items_tr),
    "n_cf_items": int(n_items),
    "sparsity_pct": float(100 * (1 - len(train) / (n_users_tr * n_items_tr))),
    "cf_sparsity_pct": float(100 * (1 - len(cf_tr) / (len(u_ids) * n_items))),
    "missing": {
        "Usia user kosong (%)": float(100 * users["Age"].isna().mean()),
        "Usia di luar 5-100 (%)": float(100 * ((users["Age"] < 5) | (users["Age"] > 100)).mean()),
        "Penulis buku kosong (%)": float(100 * (books["Book-Author"] == "").mean()),
        "Penerbit buku kosong (%)": float(100 * (books["Publisher"] == "").mean()),
    },
    "samples": {
        "ratings": _records(ratings[["User-ID", "ISBN", "Book-Rating"]].sample(8, random_state=SEED)),
        "books": _records(books[book_cols].sample(8, random_state=SEED)),
        "users": _records(users.sample(8, random_state=SEED)),
    },
}

age = users["Age"].dropna()
age = age[(age >= 5) & (age <= 100)]
year_hist = {}
if "Year-Of-Publication" in books.columns:
    yr = pd.to_numeric(books["Year-Of-Publication"], errors="coerce")
    year_hist = _hist(yr[(yr > 1900) & (yr <= 2010)], list(range(1900, 2011, 10)))

act_bins = dict(bins=[0, 1, 2, 4, 9, 19, 49, np.inf], labels=["1", "2", "3-4", "5-9", "10-19", "20-49", "50+"])
top_books = item_g.sort_values("count", ascending=False).head(20)
agg_kw = dict(n=("Book-Rating", "size"), mean=("Book-Rating", "mean"))
top_auth = (ratings[ratings["Book-Author"] != ""].groupby("Book-Author").agg(**agg_kw)
            .sort_values("n", ascending=False).head(15).reset_index())
top_pub = (ratings[ratings["Publisher"] != ""].groupby("Publisher").agg(**agg_kw)
           .sort_values("n", ascending=False).head(15).reset_index())
top_auth["Book-Author"] = top_auth["Book-Author"].str.title()
top_pub["Publisher"] = top_pub["Publisher"].str.title()

dashboard = {
    "pipeline": pipeline,
    "rating_dist": {str(k): int(v) for k, v in ratings["Book-Rating"].value_counts().sort_index().items()},
    "user_mean_hist": _hist(umean, list(range(0, 11))),
    "age_hist": _hist(age, list(range(0, 101, 10))),
    "year_hist": year_hist,
    "user_activity_hist": _hist(pd.Series(list(user_cnt.values())), **act_bins),
    "item_activity_hist": _hist(pd.Series(list(item_cnt.values())), **act_bins),
    "route_share": {k: int(v) for k, v in test["route"].value_counts().items()},
    "top_books": [{"ISBN": i, "title": meta["Book-Title"].get(i, i),
                   "author": meta["Book-Author"].get(i, "").title(),
                   "n_ratings": int(r["count"]), "mean": float(r["sum"] / r["count"])}
                  for i, r in top_books.iterrows()],
    "top_authors": _records(top_auth),
    "top_publishers": _records(top_pub),
    "rating_eval": _records(res_tbl),
    "route_eval": _records(route_tbl),
    "topk_eval": _records(topk_tbl),
}
with open(f"{ART_DIR}/dashboard_data.json", "w", encoding="utf-8") as f:
    json.dump(dashboard, f, ensure_ascii=False, indent=2)

test[["User-ID", "ISBN", "Book-Rating", "pred", "base", "route"]] \
    .sample(min(5000, len(test)), random_state=SEED) \
    .to_csv(f"{ART_DIR}/test_sample.csv", index=False)

# --- 8e. Config / metadata versi ---
config = {
    "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "seed": SEED, "test_size": TEST_SIZE,
    "MIN_ITEM_CF": MIN_ITEM_CF, "MIN_USER_CF": MIN_USER_CF, "K_NEIGHBORS": K_NEIGHBORS,
    "SHRINK": SHRINK, "REG_BIAS": REG_BIAS, "DAMP": DAMP, "TOP_K": TOP_K,
    "REL_THRESHOLD": REL_THRESHOLD, "PRUNE_TOPN": PRUNE_TOPN,
    "n_train": int(len(train)), "n_test": int(len(test)),
    "n_users": int(n_users_tr), "n_books": int(n_items_tr),
    "n_cf_items": int(n_items), "global_mean": float(mu),
}
with open(f"{ART_DIR}/config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

# --- 8f. Verifikasi: muat ulang artefak & bandingkan dengan prediksi di memori ---
try:
    from recommender_core import Recommender
    rec = Recommender(ART_DIR)
    chk = test.sample(min(300, len(test)), random_state=SEED)
    new = [rec.predict(u, i)[0] for u, i in zip(chk["User-ID"], chk["ISBN"])]
    diff = np.abs(np.array(new) - chk["pred"].values)
    log(f"Verifikasi artefak: selisih maks={diff.max():.5f}, rata-rata={diff.mean():.6f} "
        f"({'OK' if diff.max() < 1e-2 else 'PERIKSA - ada selisih (wajar bila PRUNE_TOPN diisi)'})")
except Exception as e:
    log(f"Verifikasi dilewati: {e!r}  (pastikan recommender_core.py satu folder)")

log(f"Artefak tersimpan di {ART_DIR}/ :")
for fn in sorted(os.listdir(ART_DIR)):
    print(f"   - {fn:25s} {_size(os.path.join(ART_DIR, fn))}")

# --- 8g. Zip semua artefak (mudah diunduh dari tab Output Kaggle) ---
zip_path = shutil.make_archive(os.path.join(OUT_DIR, "artifacts"), "zip", ART_DIR)
log(f"ZIP siap diunduh: {zip_path} ({_size(zip_path)})")
