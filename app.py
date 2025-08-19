# LottoLens • Lottery Stats Explorer (Educational)
# Multi-lottery: Powerball, Mega Millions, or Upload CSV.
# Features: hot/cold, pairs/triplets, 69×69 pair heatmap, EZ-pick simulation,
# countdown to next Powerball draw, auto-refresh near draws, prize mapping,
# state links, Gumroad license unlock, credits model (safe; not gambling).
#
# IMPORTANT: This app does NOT predict results. Educational/entertainment only.

import os, io, re, itertools, time
from collections import Counter
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from zoneinfo import ZoneInfo
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="LottoLens • Lottery Stats Explorer", layout="wide")

# ---------- Constants & sources ----------
ET = ZoneInfo("America/New_York")
MATRIX_CUTOFF_PB = pd.to_datetime("2015-10-04").date()   # Powerball 69/26 since Oct 2015
MATRIX_CUTOFF_MM = pd.to_datetime("2017-10-31").date()   # Mega Millions 70/25 since Oct 2017

TX_PB_CSV = "https://www.texaslottery.com/export/sites/lottery/Games/Powerball/Winning_Numbers/powerball.csv"  # official PB CSV (TX)  :contentReference[oaicite:8]{index=8}
NY_PB_CSV = "https://data.ny.gov/api/views/d6yy-54nr/rows.csv?accessType=DOWNLOAD"                              # PB NY Open Data CSV   :contentReference[oaicite:9]{index=9}
NY_MM_CSV = "https://data.ny.gov/api/views/5xaw-6ayf/rows.csv?accessType=DOWNLOAD"                              # MM NY Open Data CSV   :contentReference[oaicite:10]{index=10}

PRODUCT_ID = st.secrets.get("GUMROAD_PRODUCT_ID", "").strip()
DEMO_ONLY  = st.secrets.get("DEMO_ONLY", "false").lower() == "true"

# ---------- Licensing ----------
def verify_gumroad_license(license_key: str) -> bool:
    if not PRODUCT_ID or not license_key:
        return False
    try:
        r = requests.post("https://api.gumroad.com/v2/licenses/verify",
                          data={"product_id": PRODUCT_ID, "license_key": license_key.strip()}, timeout=20)
        js = r.json()
        return bool(js.get("success")) and not js.get("purchase", {}).get("refunded", False)
    except Exception:
        return False
# Gumroad License Keys doc: product_id required for products created on/after Jan 9, 2023.  :contentReference[oaicite:11]{index=11}

# ---------- Utility ----------
def to_table(counter, domain):
    total = sum(counter.get(n,0) for n in domain)
    rows = []
    for n in domain:
        c = int(counter.get(n,0))
        pct = (100*c/total) if total else 0
        rows.append({"number": n, "count": c, "percent": round(pct,3)})
    return pd.DataFrame(rows)

def pairs_triplets(df):
    pair_counts, trip_counts = Counter(), Counter()
    for ws in df["W"]:
        ws = sorted(ws)
        for a,b in itertools.combinations(ws, 2):
            pair_counts[(a,b)] += 1
        for a,b,c in itertools.combinations(ws, 3):
            trip_counts[(a,b,c)] += 1
    return pair_counts, trip_counts

def pair_matrix(pair_counts, max_white):
    mat = np.zeros((max_white, max_white), dtype=int)
    for (a,b),cnt in pair_counts.items():
        mat[a-1,b-1] = cnt
        mat[b-1,a-1] = cnt
    return mat

def freq_counts(df):
    whites = list(itertools.chain.from_iterable(df["W"].tolist()))
    reds = df["R"].tolist()
    return Counter(whites), Counter(reds)

def quick_pick_sim(n_tickets, max_white, max_red, seed=2025):
    rng = np.random.default_rng(seed)
    wc, rc = Counter(), Counter()
    for _ in range(n_tickets):
        whites = rng.choice(np.arange(1, max_white+1), size=5, replace=False)
        red = int(rng.integers(1, max_red+1))
        for w in whites: wc[int(w)] += 1
        rc[red] += 1
    return wc, rc

# ---------- Draw timing & auto-refresh ----------
def next_powerball_draw(now=None):
    """Next Powerball draw (Mon/Wed/Sat @ 10:59 pm ET)."""
    # Official: Powerball draws Mon/Wed/Sat 10:59 pm ET (Tallahassee).  :contentReference[oaicite:12]{index=12}
    if now is None: now = datetime.now(ET)
    draw_days = {0,2,5}  # Mon, Wed, Sat
    d = now
    while True:
        draw_dt = datetime(d.year, d.month, d.day, 22, 59, tzinfo=ET)
        if d.weekday() in draw_days and now < draw_dt:
            return draw_dt
        d += timedelta(days=1)

def get_powerball_jackpot_estimate():
    """Best-effort scrape of estimated jackpot from Powerball.com; fallback to manual input."""
    try:
        html = requests.get("https://www.powerball.com/", timeout=15).text  # :contentReference[oaicite:13]{index=13}
        m = re.search(r"Estimated Jackpot[^$]*\$\s*([\d,.]+)\s*(Million|Billion)", html, re.I)
        if m:
            num = float(m.group(1).replace(",", ""))
            mult = m.group(2).lower()
            if "billion" in mult: return int(num * 1_000_000_000)
            return int(num * 1_000_000)
    except Exception:
        pass
    return None

# ---------- Loaders (official/public) ----------
@st.cache_data(ttl=600, show_spinner=False)
def load_powerball():
    # Try Texas CSV first (fast, official)  :contentReference[oaicite:14]{index=14}
    rows=[]
    try:
        txt = requests.get(TX_PB_CSV, timeout=30).text.strip().splitlines()
        for line in txt:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 11 or "powerball" not in parts[0].lower():
                continue
            mm, dd, yy = int(parts[1]), int(parts[2]), int(parts[3])
            whites = list(map(int, parts[4:9])); red = int(parts[9])
            rows.append({"date": datetime(yy,mm,dd).date(),
                         "w1":whites[0],"w2":whites[1],"w3":whites[2],"w4":whites[3],"w5":whites[4],"r":red})
    except Exception:
        pass
    if not rows:
        # Fallback NY CSV (Open Data)  :contentReference[oaicite:15]{index=15}
        try:
            csv = pd.read_csv(NY_PB_CSV)
            for _,rec in csv.iterrows():
                try:
                    d = pd.to_datetime(str(rec.get("Draw Date") or rec.get("draw_date"))).date()
                except Exception:
                    continue
                wn = str(rec.get("Winning Numbers") or "").replace(",", " ")
                nums = [int(x) for x in wn.split() if x.isdigit()]
                if len(nums) < 6:  # need 5 whites + 1 PB
                    continue
                whites = nums[:5]; red = nums[5]
                rows.append({"date": d, "w1":whites[0],"w2":whites[1],"w3":whites[2],"w4":whites[3],"w5":whites[4],"r":int(red)})
        except Exception:
            pass
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["draw_date"] = df["date"]
    df["W"] = df[["w1","w2","w3","w4","w5"]].values.tolist()
    df["R"] = df["r"].astype(int)
    df = df[df["draw_date"] >= MATRIX_CUTOFF_PB]
    df = df[(df["W"].apply(lambda ws: len(ws)==5 and all(1<=x<=69 for x in ws))) & (df["R"].between(1,26))]
    return df.sort_values("draw_date").reset_index(drop=True)

@st.cache_data(ttl=600, show_spinner=False)
def load_megamillions():
    # NY Open Data CSV (current 70/25 matrix)  :contentReference[oaicite:16]{index=16}
    try:
        csv = pd.read_csv(NY_MM_CSV)
        rows=[]
        for _,rec in csv.iterrows():
            try:
                d = pd.to_datetime(str(rec.get("Draw Date") or rec.get("draw_date"))).date()
            except Exception:
                continue
            wn = str(rec.get("Winning Numbers") or "").replace(",", " ")
            nums = [int(x) for x in wn.split() if x.isdigit()]
            if len(nums) < 6:  # need 5 whites + 1 mega ball
                continue
            whites = nums[:5]; mega = nums[5]
            rows.append({"date": d, "w1":whites[0],"w2":whites[1],"w3":whites[2],"w4":whites[3],"w5":whites[4],"r":int(mega)})
        df = pd.DataFrame(rows)
    except Exception:
        df = pd.DataFrame()
    if df.empty:
        return df
    df["draw_date"] = df["date"]
    df["W"] = df[["w1","w2","w3","w4","w5"]].values.tolist()
    df["R"] = df["r"].astype(int)
    df = df[df["draw_date"] >= MATRIX_CUTOFF_MM]
    df = df[(df["W"].apply(lambda ws: len(ws)==5 and all(1<=x<=70 for x in ws))) & (df["R"].between(1,25))]
    return df.sort_values("draw_date").reset_index(drop=True)

# ---------- Prize mapping (Powerball tiers) ----------
# Official prize chart (without Power Play).  :contentReference[oaicite:17]{index=17}
POWERBALL_PRIZES = {
    (5, True):  "JACKPOT",   # 5 + PB
    (5, False): 1_000_000,
    (4, True):  50_000,
    (4, False): 100,
    (3, True):  100,
    (3, False): 7,
    (2, True):  7,
    (1, True):  4,
    (0, True):  4
}

def prize_for_result(match_white, pb_matched, jackpot_estimate):
    val = POWERBALL_PRIZES.get((match_white, pb_matched), 0)
    return jackpot_estimate if val == "JACKPOT" else val

# ---------- Sidebar: licensing & options ----------
st.sidebar.title("Access")
license_key = st.sidebar.text_input("Enter Gumroad license key", type="password")
licensed = verify_gumroad_license(license_key)
if DEMO_ONLY:
    licensed = False
st.sidebar.write("License:", "✅ Valid" if licensed else "🔓 Demo mode")

st.sidebar.title("Choose Lottery")
lottery = st.sidebar.selectbox("Edition", ["Powerball (US)", "Mega Millions (US)", "Upload CSV (any game)"])

# Credits (safe, not gambling)
st.sidebar.title("Credits")
if "credits" not in st.session_state:
    st.session_state.credits = 60  # free starter credits
st.sidebar.write(f"Available credits: {st.session_state.credits}")
def spend_credits(n=1):
    if st.session_state.credits >= n:
        st.session_state.credits -= n
        return True
    st.warning("Not enough credits. Upgrade to Pro for more.")
    return False

# ---------- Title & Disclaimer ----------
st.title("LottoLens • Lottery Stats Explorer")
with st.expander("Read me first (disclaimer)"):
    st.markdown("""
**DISCLAIMER:** This app is for **educational and entertainment** purposes only.  
It **does not predict** or improve your chances of winning any lottery.  
Lotteries are games of chance. Use at your own risk.  
Not affiliated with Powerball, Mega Millions, or any lottery commission.
""")

# ---------- Countdown & auto-refresh (Powerball only block shown to everyone) ----------
st.subheader("Next Powerball draw")
npd = next_powerball_draw()
left = npd - datetime.now(ET)
interval_ms = 30000 if left.total_seconds() > 900 else 10000
st_autorefresh(interval=interval_ms, key="auto_refresh")

colA, colB = st.columns(2)
with colA: st.metric("Draw time (ET)", npd.strftime("%a %b %d, %Y 10:59 pm ET"))
with colB:
    secs = int(max(left.total_seconds(), 0))
    h = secs // 3600; m = (secs % 3600) // 60; s = secs % 60
    st.metric("Countdown", f"{h:02d}:{m:02d}:{s:02d}")
st.caption("Official draw time: 10:59 pm ET (Mon/Wed/Sat). Watch on Powerball.com. ")  # :contentReference[oaicite:18]{index=18}

# ---------- Load data per lottery ----------
if lottery.startswith("Powerball"):
    df = load_powerball()
    white_max, red_max = 69, 26
elif lottery.startswith("Mega"):
    df = load_megamillions()
    white_max, red_max = 70, 25
else:
    st.subheader("Upload your history CSV")
    st.write("CSV columns required: **Date, White1..White5, BonusBall** (6th optional for games without bonus).")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up:
        raw = pd.read_csv(up)
        rows=[]
        for _,r in raw.iterrows():
            try:
                d = pd.to_datetime(str(r["Date"])).date()
                whites = [int(r["White1"]),int(r["White2"]),int(r["White3"]),int(r["White4"]),int(r["White5"])]
                bonus = int(r.get("BonusBall")) if "BonusBall" in r else 0
                rows.append({"draw_date": d, "W": sorted(whites), "R": bonus})
            except Exception:
                pass
        df = pd.DataFrame(rows).sort_values("draw_date").reset_index(drop=True)
        white_max = int(max(itertools.chain.from_iterable(df["W"]))) if not df.empty else 69
        red_max = int(max(df["R"])) if not df.empty else 26
    else:
        df = pd.DataFrame(columns=["draw_date","W","R"])
        white_max, red_max = 69, 26

if df.empty:
    st.error("Could not load historical data. Try again later or use the Upload option.")
    st.stop()

# In demo, limit rows
df_show = df.copy()
if not licensed:
    df_show = df.tail(100)
    st.info("Demo mode: last 100 draws only. Enter a valid license key to unlock full history, downloads, and all charts.")

# Manual refresh
if st.button("🔁 Refresh now"):
    st.cache_data.clear()
    st.experimental_rerun()

# ---------- Headline cards ----------
c1,c2,c3 = st.columns(3)
with c1: st.metric("Draws (shown)", len(df_show))
with c2: st.metric("Date range", f"{df_show['draw_date'].min()} → {df_show['draw_date'].max()}")
with c3: st.metric("Edition", lottery)

# ---------- Frequencies ----------
wc_all, rc_all = freq_counts(df_show)
white_tbl = to_table(wc_all, range(1, white_max+1)).sort_values(["count","number"], ascending=[False,True])
red_tbl   = to_table(rc_all, range(1, red_max+1)).sort_values(["count","number"], ascending=[False,True])

st.subheader("Overall frequencies")
x1, x2 = st.columns([2,1])
with x1:
    fig, ax = plt.subplots(figsize=(12,4))
    sns.barplot(x="number", y="count", data=white_tbl.sort_values("number"), ax=ax, palette="viridis")
    ax.set_title("White-ball counts"); ax.set_xlabel("Number"); ax.set_ylabel("Count")
    st.pyplot(fig)
with x2:
    fig2, ax2 = plt.subplots(figsize=(6,4))
    sns.barplot(x="number", y="count", data=red_tbl.sort_values("number"), ax=ax2, palette="rocket")
    ax2.set_title("Bonus-ball counts"); ax2.set_xlabel("Bonus"); ax2.set_ylabel("Count")
    st.pyplot(fig2)

# ---------- Recent hot/cold ----------
tail_n = st.sidebar.select_slider("Recent window (hot/cold)", options=[50,100,250,500], value=100)
wc_tail, rc_tail = freq_counts(df_show.tail(min(tail_n, len(df_show))))
st.subheader(f"Recent hot/cold (last {min(tail_n, len(df_show))} draws)")
y1,y2 = st.columns(2)
with y1:
    st.write("**Hot whites (top 10)**")
    st.dataframe(to_table(wc_tail, range(1, white_max+1)).sort_values("count", ascending=False).head(10), use_container_width=True)
with y2:
    st.write("**Cold whites (bottom 10)**")
    st.dataframe(to_table(wc_tail, range(1, white_max+1)).sort_values("count", ascending=True).head(10), use_container_width=True)

# ---------- Pairs & Triplets (cost 1 credit to generate charts) ----------
st.subheader("Common pairs & triplets (whites)")
if st.button("Generate pairs/triplets (cost: 1 credit)"):
    if spend_credits(1):
        pair_counts, trip_counts = pairs_triplets(df_show)
        z1,z2 = st.columns(2)
        with z1:
            st.dataframe(pd.DataFrame(pair_counts.most_common(15), columns=["pair","count"]), use_container_width=True)
        with z2:
            st.dataframe(pd.DataFrame(trip_counts.most_common(15), columns=["triplet","count"]), use_container_width=True)
        # Heatmap
        st.write("**Pair heatmap (69×69 style)**")
        mat = pair_matrix(pair_counts, white_max)
        fig3, ax3 = plt.subplots(figsize=(10,8))
        sns.heatmap(mat, cmap="YlOrRd", cbar=True, ax=ax3)
        ax3.set_title("Pair frequency heatmap"); ax3.set_xlabel("White ball"); ax3.set_ylabel("White ball")
        st.pyplot(fig3)

# ---------- EZ-pick simulation ----------
sim_n = st.sidebar.select_slider("EZ-pick simulation tickets", options=[1000,5000,10000,20000], value=10000)
st.subheader(f"EZ-pick simulation — {sim_n:,} random tickets")
sim_wc, sim_rc = quick_pick_sim(sim_n, white_max, red_max)
sim_tbl = to_table(sim_wc, range(1, white_max+1)).sort_values("number")
fig4, ax4 = plt.subplots(figsize=(12,4))
sns.barplot(x="number", y="count", data=sim_tbl, ax=ax4, palette="coolwarm")
ax4.set_title("Simulated white-ball pick counts"); ax4.set_xlabel("Number"); ax4.set_ylabel("Count")
st.pyplot(fig4)

# ---------- Prize mapping demo (Powerball only) ----------
if lottery.startswith("Powerball"):
    st.subheader("Simulated prize mapping (official tiers)")
    jackpot_est = get_powerball_jackpot_estimate()
    if not jackpot_est:
        jackpot_est = st.number_input("Estimated Jackpot (USD)", min_value=20_000_000, value=500_000_000, step=5_000_000)
    st.caption("Prize tiers per official chart; jackpot is an estimate and may change.")  # :contentReference[oaicite:19]{index=19}

    # Tiny demo: evaluate 5 random tickets vs last draw
    if len(df_show) >= 1:
        last_w = set(df_show.iloc[-1]["W"])
        last_r = df_show.iloc[-1]["R"]
        st.write(f"Last draw shown: {df_show.iloc[-1]['draw_date']} — whites {sorted(last_w)}, bonus {last_r}")
        rng = np.random.default_rng(123)
        demo_rows=[]
        for _ in range(5):
            ws = sorted(rng.choice(np.arange(1,70), size=5, replace=False))
            r  = int(rng.integers(1,27))
            mw = len(set(ws) & last_w)
            prize = prize_for_result(mw, r==last_r, jackpot_est)
            demo_rows.append({"ticket_whites":ws, "ticket_bonus":r, "match_whites":mw, "match_bonus":(r==last_r), "prize_usd":prize})
        st.table(pd.DataFrame(demo_rows))

# ---------- State rules & links (informational) ----------
st.subheader("State rules & purchase cut-offs (informational, not legal advice)")
state = st.selectbox("Your state", sorted([
    "Alabama (N/A)","Arizona","California","Florida","Michigan","New York","Texas","Washington DC","Puerto Rico","US Virgin Islands"
]))
st.write("• Official Powerball site (draw times & FAQs): https://www.powerball.com/")  # Sales cut-offs vary by jurisdiction  :contentReference[oaicite:20]{index=20}
if state == "Florida":
    st.write("• Florida: Draw 10:59 pm ET; ticket sales cut-off 10:00 pm ET (official). https://floridalottery.com/games/draw-games/powerball")  # :contentReference[oaicite:21]{index=21}
if state == "Michigan":
    st.write("• Michigan: Draw 10:59 pm ET (official schedule). Online sales show 9:58 pm ET cutoff note. https://faq.michiganlottery.com/... / https://www.michiganlottery.com/games/powerball")  # :contentReference[oaicite:22]{index=22}
st.info("Sales cut-off times vary by 1–2 hours before the draw depending on jurisdiction. Always check your state’s official lottery site.")  # :contentReference[oaicite:23]{index=23}

# ---------- Downloads (licensed only) ----------
if licensed:
    st.subheader("Download tables (CSV)")
    st.download_button("White frequencies CSV", white_tbl.to_csv(index=False), "whites.csv", "text/csv")
    st.download_button("Bonus frequencies CSV", red_tbl.to_csv(index=False), "bonus.csv", "text/csv")
    out_df = df_show.copy()
    out_df["W1"] = [w[0] for w in out_df["W"]]; out_df["W2"] = [w[1] for w in out_df["W"]]
    out_df["W3"] = [w[2] for w in out_df["W"]]; out_df["W4"] = [w[3] for w in out_df["W"]]; out_df["W5"] = [w[4] for w in out_df["W"]]
    st.download_button("Draw history (shown) CSV", out_df.drop(columns=["W"]).to_csv(index=False), "history.csv", "text/csv")

st.caption("Data: Texas Lottery CSV; NY Open Data. LottoLens is not affiliated with any lottery. Educational/entertainment only.")
