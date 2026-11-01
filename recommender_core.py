"""
recommender_core.py
===================
Memuat artefak hasil book_recommender.py (folder `artifacts/`) dan menyediakan
inferensi Switching Hybrid (CF -> Content -> Rule/Baseline) untuk Streamlit.
"""
import json
import pickle

import numpy as np
import pandas as pd
from scipy import sparse


class Recommender:
    def __init__(self, art_dir="artifacts"):
        self.cfg = json.load(open(f"{art_dir}/config.json", encoding="utf-8"))
        with open(f"{art_dir}/model_state.pkl", "rb") as f:
            st = pickle.load(f)
        self.mu = st["mu"]
        self.b_i, self.b_u = st["b_i"], st["b_u"]
        self.user_cnt, self.item_cnt = st["user_cnt"], st["item_cnt"]
        self.cf_items, self.item_ix = st["cf_items"], st["item_ix"]
        self.user_hist = st["user_hist"]
        self.auth_d, self.pub_d = st["auth_d"], st["pub_d"]
        self.pop_rank, self.pop_score = st["pop_rank"], st["pop_score"]
        self.author_items = st["author_items"]
        self.item_arr = np.array(self.cf_items)

        self.S = sparse.load_npz(f"{art_dir}/similarity.npz").tocsr()
        self.books = pd.read_csv(f"{art_dir}/books.csv.gz", dtype=str,
                                 keep_default_na=False).set_index("ISBN")
        self.history = pd.read_csv(f"{art_dir}/user_history.csv.gz",
                                   dtype={"User-ID": int, "ISBN": str, "Book-Rating": int})
        self.seen = self.history.groupby("User-ID")["ISBN"].apply(set).to_dict()
        self._search = (self.books["Book-Title"] + " " + self.books["Book-Author"]).str.lower()

        self.min_user_cf = self.cfg["MIN_USER_CF"]
        self.min_item_cf = self.cfg["MIN_ITEM_CF"]
        self.k_nb = self.cfg["K_NEIGHBORS"]
        self.damp = self.cfg["DAMP"]
        self.rel = self.cfg["REL_THRESHOLD"]

    # ------------------------------------------------------------------ util
    def baseline(self, u, i):
        return self.mu + self.b_i.get(i, 0.0) + self.b_u.get(u, 0.0)

    def title(self, i):
        return self.books["Book-Title"].get(i, i)

    def author(self, i):
        return self.books["Author-Display"].get(i, "")

    def _meta(self, i):
        if i in self.books.index:
            r = self.books.loc[i]
            return r["Book-Author"], r["Publisher"]
        return "", ""

    def label(self, i):
        return f"{self.title(i)} — {self.author(i)}  ({self.item_cnt.get(i, 0)} rating)"

    # ------------------------------------------------------------ prediksi
    def _cf_core(self, u, i):
        if i not in self.item_ix or u not in self.user_hist:
            return None
        js, res, _ = self.user_hist[u]
        row = self.S.getrow(self.item_ix[i]).toarray().ravel()
        s = row[js]
        pos = s > 0
        if not pos.any():
            return None
        s, res, js = s[pos], res[pos], js[pos]
        if len(s) > self.k_nb:
            top = np.argpartition(-s, self.k_nb)[:self.k_nb]
            s, res, js = s[top], res[top], js[top]
        p = self.baseline(u, i) + float(np.dot(s, res) / (np.abs(s).sum() + self.damp))
        return p, s, res, js

    def _cb(self, u, i, author, pub):
        for d, key in ((self.auth_d, (u, author)), (self.pub_d, (u, pub))):
            if key[1] and key in d:
                s, c = d[key]
                return self.baseline(u, i) + (s / c) * (c / (c + 2.0)), key
        return None

    def predict(self, u, i, author=None, pub=None):
        """Return (prediksi rating 1-10, route)."""
        d = self.predict_detail(u, i, author, pub)
        return d["pred"], d["route"]

    def predict_detail(self, u, i, author=None, pub=None):
        """Prediksi + rincian (komponen baseline, tetangga CF / sinyal content)."""
        if author is None or pub is None:
            author, pub = self._meta(i)
        out = {"mu": self.mu, "b_i": self.b_i.get(i, 0.0), "b_u": self.b_u.get(u, 0.0),
               "baseline": float(np.clip(self.baseline(u, i), 1, 10)),
               "neighbors": None, "note": ""}
        uc, ic = self.user_cnt.get(u, 0), self.item_cnt.get(i, 0)
        if uc >= self.min_user_cf and ic >= self.min_item_cf:
            r = self._cf_core(u, i)
            if r is not None:
                p, s, res, js = r
                nb = pd.DataFrame({
                    "ISBN": [self.item_arr[j] for j in js],
                    "Kemiripan": s.astype(float),
                    "Residual rating user": res.astype(float)})
                nb.insert(0, "Judul", [self.title(x) for x in nb["ISBN"]])
                out.update(pred=float(np.clip(p, 1, 10)), route="CF",
                           neighbors=nb.sort_values("Kemiripan", ascending=False).reset_index(drop=True),
                           note="Baseline + rata-rata terbobot residual dari buku mirip yang pernah dinilai user.")
                return out
        if uc > 0:
            r = self._cb(u, i, author, pub)
            if r is not None:
                p, key = r
                what = "penulis" if key[1] == author else "penerbit"
                out.update(pred=float(np.clip(p, 1, 10)), route="Content",
                           note=f"Baseline + kecenderungan user pada {what} yang sama: '{key[1]}'.")
                return out
        out.update(pred=out["baseline"], route="Rule/Baseline",
                   note="Tidak ada sinyal CF/Content -> memakai baseline (rata-rata global + bias user + bias buku).")
        return out

    # ------------------------------------------------------- top-K rekomendasi
    def _rec_pop(self, u):
        seen = self.seen.get(u, set())
        return [i for i in self.pop_rank if i not in seen]

    def _score_cf(self, js, w, seen, k):
        """js: indeks item (di S) yang dinilai, w: bobot (rating/10). -> [(isbn, skor, alasan)]"""
        js = np.asarray(js)
        sub = self.S[:, js].toarray()
        sc = sub @ w
        sc[js] = -1
        m = min(k + 5, len(sc) - 1)
        if m < 1:
            return []
        top = np.argpartition(-sc, m)[:m]
        top = top[np.argsort(-sc[top])]
        out = []
        for t in top:
            isbn = self.item_arr[t]
            if sc[t] <= 0 or isbn in seen:
                continue
            contrib = sub[t] * w
            order = [o for o in np.argsort(-contrib)[:2] if contrib[o] > 0]
            src = "; ".join(self.title(self.item_arr[js[o]]) for o in order)
            out.append((isbn, float(sc[t]), f"Mirip dengan: {src}"))
        return out[:k]

    def _rec_cf(self, u, k):
        js, _, rating = self.user_hist[u]
        return self._score_cf(js, rating / 10.0, self.seen.get(u, set()), k)

    def _rec_content(self, u, k):
        h = self.history[(self.history["User-ID"] == u) & (self.history["Book-Rating"] >= self.rel)]
        seen, have, out = self.seen.get(u, set()), set(), []
        for i in h["ISBN"]:
            a = self.books["Book-Author"].get(i, "")
            for j in self.author_items.get(a, [])[:5]:
                if j not in seen and j not in have:
                    out.append((j, 0.0, f"Penulis sama dengan '{self.title(i)}'"))
                    have.add(j)
        return out[:k]

    def _fill_pop(self, items, pool, k):
        have = {x[0] for x in items}
        for i in pool:
            if len(items) >= k:
                break
            if i not in have:
                items.append((i, 0.0, "Populer (pelengkap)"))
                have.add(i)
        return items

    def recommend(self, u, k=10):
        """Return (DataFrame rekomendasi, route yang dipakai)."""
        uc = self.user_cnt.get(u, 0)
        if uc >= self.min_user_cf and u in self.user_hist:
            items, route = self._rec_cf(u, k), "CF"
        elif uc > 0:
            items, route = self._rec_content(u, k), "Content"
        else:
            items, route = [], "Rule/Popularitas"
        items = self._fill_pop(items, self._rec_pop(u), k)
        return self._frame(items), route

    def recommend_profile(self, ratings, k=10):
        """Cold-start interaktif: ratings = {isbn: rating} dari user baru."""
        usable = [i for i in ratings if i in self.item_ix]
        items, route = [], "Rule/Popularitas"
        if usable:
            js = [self.item_ix[i] for i in usable]
            w = np.array([ratings[i] / 10.0 for i in usable])
            items = self._score_cf(js, w, set(ratings), k)
            if items:
                route = "CF (profil baru)"
        pool = [i for i in self.pop_rank if i not in ratings]
        return self._frame(self._fill_pop(items, pool, k)), route

    def similar_items(self, isbn, k=10):
        if isbn not in self.item_ix:
            return self._frame([])
        row = self.S.getrow(self.item_ix[isbn]).toarray().ravel()
        top = [t for t in np.argsort(-row)[:k] if row[t] > 0]
        return self._frame([(self.item_arr[t], float(row[t]), "Kemiripan item-item") for t in top])

    def popular(self, k=10, author=None, publisher=None, year_range=None, min_ratings=3):
        df = pd.DataFrame({"ISBN": list(self.pop_score), "score": list(self.pop_score.values())})
        df["n"] = df["ISBN"].map(self.item_cnt)
        df = df[df["n"] >= min_ratings].merge(self.books.reset_index(), on="ISBN", how="left")
        if author:
            df = df[df["Author-Display"] == author]
        if publisher:
            df = df[df["Publisher"] == publisher]
        if year_range and "Year-Of-Publication" in df.columns:
            y = pd.to_numeric(df["Year-Of-Publication"], errors="coerce")
            df = df[(y >= year_range[0]) & (y <= year_range[1])]
        df = df.sort_values(["score", "n"], ascending=False).head(k)
        return self._frame([(r.ISBN, float(r.score), "Skor popularitas ter-damping") for r in df.itertuples()])

    # ------------------------------------------------------------ pencarian
    def search_books(self, q="", n=30, only_cf=False):
        s = self._search
        s = s[s.index.isin(self.item_ix if only_cf else self.item_cnt)]
        q = q.strip().lower()
        if q:
            s = s[s.str.contains(q, regex=False)]
        cnt = np.array([self.item_cnt.get(i, 0) for i in s.index])
        return list(s.index[np.argsort(-cnt)[:n]])

    def top_values(self, col, n=300):
        cnt = self.books.index.map(lambda i: self.item_cnt.get(i, 0))
        g = self.books.assign(_n=np.asarray(cnt)).groupby(col)["_n"].sum().sort_values(ascending=False)
        return [v for v in g.index[:n] if v]

    # --------------------------------------------------------------- tampilan
    def _frame(self, items):
        cols = ["Book-Title", "Author-Display", "Publisher", "Year-Of-Publication", "Image-URL-M"]
        cols = [c for c in cols if c in self.books.columns]
        isbns = [x[0] for x in items]
        df = self.books.reindex(isbns)[cols].fillna("").copy()
        df.index = pd.Index(isbns, name="ISBN")
        df = df.reset_index()
        df["Jml rating"] = df["ISBN"].map(self.item_cnt).fillna(0).astype(int)
        df["Skor"] = [x[1] for x in items]
        df["Alasan"] = [x[2] for x in items]
        return df

    def user_history(self, u, n=10):
        h = self.history[self.history["User-ID"] == u].sort_values("Book-Rating", ascending=False).head(n)
        return h.merge(self.books[["Book-Title", "Author-Display"]],
                       left_on="ISBN", right_index=True, how="left")
