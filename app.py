import streamlit as st
import pandas as pd
import numpy as np
import warnings, io, pickle
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction · ML Notebook",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=IBM+Plex+Mono:wght@300;400;500&display=swap');

:root {
  --bg:      #f9f8f5;
  --surface: #ffffff;
  --border:  #ece9e3;
  --teal:    #0d9488;
  --teal2:   #ccfbf1;
  --amber:   #d97706;
  --amber2:  #fef3c7;
  --red:     #dc2626;
  --red2:    #fee2e2;
  --blue:    #2563eb;
  --blue2:   #dbeafe;
  --green:   #16a34a;
  --green2:  #dcfce7;
  --ink:     #1c1917;
  --ink2:    #57534e;
  --ink3:    #a8a29e;
  --shadow:  0 1px 8px rgba(0,0,0,.07);
  --shadow2: 0 4px 20px rgba(0,0,0,.09);
}

html,body,[class*="css"]  { font-family:'Outfit',sans-serif!important; background:var(--bg)!important; color:var(--ink)!important; }
.stApp                    { background:var(--bg)!important; }
.block-container          { padding:2rem 2.5rem!important; max-width:1200px!important; }

/* ── Top header ── */
.nb-header {
  display:flex; align-items:center; justify-content:space-between;
  padding:1.6rem 2rem; background:var(--surface); border:1px solid var(--border);
  border-radius:18px; margin-bottom:1.6rem; box-shadow:var(--shadow);
}
.nb-title  { font-size:1.9rem; font-weight:800; color:var(--ink); line-height:1.1; }
.nb-title span { color:var(--teal); }
.nb-sub    { font-size:.82rem; color:var(--ink3); margin-top:.3rem; font-weight:400; }
.nb-badge  {
  background:var(--teal2); color:var(--teal); border:1px solid #99f6e4;
  border-radius:999px; font-family:'IBM Plex Mono',monospace;
  font-size:.65rem; font-weight:600; letter-spacing:.07em;
  text-transform:uppercase; padding:.3rem .9rem;
}

/* ── Section title ── */
.sec {
  display:flex; align-items:center; gap:.7rem;
  font-size:.68rem; font-weight:700; text-transform:uppercase;
  letter-spacing:.14em; color:var(--ink3);
  margin:1.8rem 0 .9rem;
}
.sec::after { content:''; flex:1; height:1px; background:var(--border); }
.sec-icon {
  background:var(--teal); color:#fff; border-radius:7px;
  width:24px; height:24px; display:flex; align-items:center;
  justify-content:center; font-size:.72rem; flex-shrink:0;
}

/* ── White card ── */
.card {
  background:var(--surface); border:1px solid var(--border);
  border-radius:14px; padding:1.4rem 1.6rem;
  box-shadow:var(--shadow); margin-bottom:1rem;
}

/* ── Stat row ── */
.stat-row { display:grid; grid-template-columns:repeat(4,1fr); gap:.9rem; margin:.2rem 0 1.4rem; }
.stat {
  background:var(--surface); border:1px solid var(--border); border-radius:14px;
  padding:1.1rem 1.3rem; box-shadow:var(--shadow);
  border-left:4px solid var(--teal);
}
.stat.amber { border-left-color:var(--amber); }
.stat.red   { border-left-color:var(--red);   }
.stat.blue  { border-left-color:var(--blue);  }
.stat-lbl   { font-size:.66rem; font-weight:700; text-transform:uppercase; letter-spacing:.09em; color:var(--ink3); }
.stat-val   { font-size:1.75rem; font-weight:800; color:var(--ink); line-height:1.1; margin-top:.15rem; }
.stat-note  { font-size:.67rem; color:var(--ink3); margin-top:.12rem; }

/* ── Code-style box ── */
.codebox {
  background:#f5f3f0; border:1px solid var(--border); border-radius:10px;
  padding:.9rem 1.2rem; font-family:'IBM Plex Mono',monospace; font-size:.78rem;
  color:var(--ink2); line-height:1.7; margin:.4rem 0;
}
.codebox .kw  { color:var(--teal);  font-weight:600; }
.codebox .cm  { color:var(--ink3);  font-style:italic; }
.codebox .str { color:var(--amber); }

/* ── Step items ── */
.step-list { display:flex; flex-direction:column; gap:.5rem; margin:.2rem 0; }
.step-item {
  background:var(--surface); border:1px solid var(--border); border-radius:10px;
  padding:.8rem 1.1rem; display:flex; align-items:flex-start; gap:.85rem;
  box-shadow:var(--shadow);
}
.step-n {
  background:var(--teal); color:#fff; border-radius:7px;
  min-width:26px; height:26px; display:flex; align-items:center;
  justify-content:center; font-size:.73rem; font-weight:700; flex-shrink:0; margin-top:.05rem;
}
.step-t { font-size:.87rem; font-weight:600; color:var(--ink); }
.step-d { font-size:.75rem; color:var(--ink3); margin-top:.1rem; line-height:1.45; }

/* ── Model result card ── */
.model-wrap {
  background:var(--surface); border:1px solid var(--border); border-radius:16px;
  padding:1.5rem 1.7rem; box-shadow:var(--shadow); margin-bottom:1.1rem;
}
.model-head { display:flex; align-items:center; gap:.8rem; margin-bottom:1.1rem; }
.model-dot  { width:12px; height:12px; border-radius:50%; flex-shrink:0; }
.model-name { font-size:1.05rem; font-weight:700; color:var(--ink); }
.model-tag  {
  font-family:'IBM Plex Mono',monospace; font-size:.62rem;
  background:#f5f3f0; border:1px solid var(--border); border-radius:5px;
  color:var(--ink3); padding:.1rem .45rem; margin-left:auto;
}

/* ── Metric bar ── */
.mbar-wrap { margin:.45rem 0; }
.mbar-row  { display:flex; justify-content:space-between; align-items:center; margin-bottom:.22rem; }
.mbar-lbl  { font-size:.77rem; font-weight:500; color:var(--ink2); }
.mbar-val  { font-family:'IBM Plex Mono',monospace; font-size:.77rem; font-weight:600; color:var(--ink); }
.mbar-bg   { background:#f0ede8; border-radius:999px; height:6px; overflow:hidden; }
.mbar-fill { height:100%; border-radius:999px; }

/* ── Confusion matrix ── */
.cm-grid {
  display:grid; grid-template-columns:auto 1fr 1fr;
  gap:5px; margin-top:.8rem; font-family:'IBM Plex Mono',monospace;
}
.cm-hd  { background:#f5f3f0; color:var(--ink3); font-size:.6rem; font-weight:600;
          text-transform:uppercase; letter-spacing:.07em; border-radius:7px;
          padding:.45rem .6rem; text-align:center; }
.cm-lbl { display:flex; align-items:center; justify-content:center;
          font-size:.6rem; font-weight:600; color:var(--ink3); text-transform:uppercase;
          letter-spacing:.06em; writing-mode:horizontal-tb; padding:.3rem .4rem; }
.cm-cell { border-radius:10px; padding:.8rem .5rem; text-align:center; }
.cm-cell .big   { font-size:1.4rem; font-weight:800; display:block; line-height:1; }
.cm-cell .small { font-size:.58rem; font-weight:600; text-transform:uppercase; letter-spacing:.06em; margin-top:3px; display:block; opacity:.75; }
.cm-TP { background:var(--green2); color:var(--green); }
.cm-TN { background:var(--blue2);  color:var(--blue);  }
.cm-FP { background:var(--red2);   color:var(--red);   }
.cm-FN { background:var(--amber2); color:var(--amber); }

/* ── Oversampling viz ── */
.os-bar-wrap { display:flex; align-items:center; gap:.9rem; margin:.4rem 0; }
.os-bar-lbl  { font-size:.8rem; font-weight:600; color:var(--ink2); min-width:70px; }
.os-bar-bg   { flex:1; background:#f0ede8; border-radius:999px; height:10px; overflow:hidden; }
.os-bar-fill { height:100%; border-radius:999px; }
.os-bar-cnt  { font-family:'IBM Plex Mono',monospace; font-size:.78rem; font-weight:600; color:var(--ink); min-width:50px; text-align:right; }

/* ── Feature importance ── */
.fi-row { display:flex; align-items:center; gap:.8rem; margin:.35rem 0; }
.fi-rank { font-family:'IBM Plex Mono',monospace; font-size:.7rem; color:var(--ink3); min-width:20px; }
.fi-name { font-size:.82rem; font-weight:500; color:var(--ink2); min-width:160px; }
.fi-bg   { flex:1; background:#f0ede8; border-radius:999px; height:8px; overflow:hidden; }
.fi-fill { height:100%; border-radius:999px; background:linear-gradient(90deg,var(--teal),#14b8a6); }
.fi-score { font-family:'IBM Plex Mono',monospace; font-size:.75rem; font-weight:600; color:var(--ink); min-width:45px; text-align:right; }

/* ── Info box ── */
.infobox {
  background:var(--teal2); border:1px solid #99f6e4; border-radius:10px;
  padding:.8rem 1rem; font-size:.8rem; color:#0f766e; margin:.5rem 0;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background:var(--surface)!important; border-radius:12px!important; border:1px solid var(--border)!important; padding:.3rem!important; gap:.25rem!important; margin-bottom:1.2rem!important; }
.stTabs [data-baseweb="tab"]      { border-radius:9px!important; font-weight:600!important; font-size:.82rem!important; color:var(--ink3)!important; padding:.45rem 1rem!important; }
.stTabs [aria-selected="true"]    { background:var(--teal)!important; color:#fff!important; }

/* ── Buttons ── */
.stButton>button { background:var(--teal)!important; color:#fff!important; border:none!important; border-radius:10px!important; font-family:'Outfit',sans-serif!important; font-weight:700!important; font-size:.84rem!important; padding:.5rem 1.8rem!important; box-shadow:0 2px 10px rgba(13,148,136,.3)!important; transition:all .15s!important; }
.stButton>button:hover { opacity:.88!important; transform:translateY(-1px)!important; }

/* ── Inputs ── */
.stSelectbox label,.stSlider label,.stNumberInput label,.stFileUploader label { font-size:.72rem!important; font-weight:700!important; color:var(--ink3)!important; text-transform:uppercase!important; letter-spacing:.07em!important; }
div[data-baseweb="select"]>div { background:#f9f8f5!important; border-color:var(--border)!important; border-radius:9px!important; }

/* ── Dataframe ── */
.stDataFrame { border-radius:12px!important; overflow:hidden!important; border:1px solid var(--border)!important; }

/* ── Download ── */
.stDownloadButton>button { background:#fff!important; color:var(--teal)!important; border:1.5px solid var(--teal)!important; border-radius:10px!important; font-weight:700!important; font-size:.83rem!important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Demo data
# ─────────────────────────────────────────────
@st.cache_data
def make_demo():
    np.random.seed(7); n=1500
    data = dict(
        customerID=[f'CUST-{i:04d}' for i in range(n)],
        gender=np.random.choice(['Male','Female'],n),
        SeniorCitizen=np.random.choice([0,1],n,p=[.84,.16]),
        Partner=np.random.choice(['Yes','No'],n),
        Dependents=np.random.choice(['Yes','No'],n,p=[.3,.7]),
        tenure=np.random.randint(0,72,n),
        PhoneService=np.random.choice(['Yes','No'],n,p=[.9,.1]),
        MultipleLines=np.random.choice(['Yes','No','No phone service'],n),
        InternetService=np.random.choice(['DSL','Fiber optic','No'],n),
        OnlineSecurity=np.random.choice(['Yes','No','No internet service'],n),
        OnlineBackup=np.random.choice(['Yes','No','No internet service'],n),
        DeviceProtection=np.random.choice(['Yes','No','No internet service'],n),
        TechSupport=np.random.choice(['Yes','No','No internet service'],n),
        StreamingTV=np.random.choice(['Yes','No','No internet service'],n),
        StreamingMovies=np.random.choice(['Yes','No','No internet service'],n),
        Contract=np.random.choice(['Month-to-month','One year','Two year'],n,p=[.55,.25,.20]),
        PaperlessBilling=np.random.choice(['Yes','No'],n),
        PaymentMethod=np.random.choice(['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)'],n),
        MonthlyCharges=np.round(np.random.uniform(20,120,n),2),
        TotalCharges='',
        Churn=np.random.choice(['Yes','No'],n,p=[.265,.735]),
    )
    df=pd.DataFrame(data)
    df['TotalCharges']=(df['tenure']*df['MonthlyCharges']+np.random.normal(0,50,n)).round(2).astype(str)
    df.loc[np.random.choice(df.index,11,replace=False),'TotalCharges']=' '
    return df


def run_pipeline(df):
    """Exact notebook pipeline → returns all intermediate results"""
    ch = df.copy()

    # Cell 4 – nulls before cleaning
    nulls_before = ch.isnull().sum()[ch.isnull().sum()>0]

    # Cell 6 – object columns
    obj_cols = ch.select_dtypes(include=['object']).copy()

    # Cell 8 – ordinal encode Contract
    ch['Contract'] = ch['Contract'].replace({'Month-to-month':0,'One year':1,'Two year':2})

    # Cell 9 – drop customerID
    ch = ch.drop(['customerID'], axis=1)

    # Cell 12 – fix TotalCharges spaces
    ch['TotalCharges'] = ch['TotalCharges'].replace({' ':np.nan,'':np.nan})
    tc_nulls = int(ch['TotalCharges'].isnull().sum())
    ch['TotalCharges'] = pd.to_numeric(ch['TotalCharges'], errors='coerce')
    ch['TotalCharges'].fillna(ch['TotalCharges'].median(), inplace=True)

    # Encode Churn target
    if ch['Churn'].dtype == object:
        ch['Churn'] = ch['Churn'].map({'Yes':1,'No':0})

    # Cell 15 – label encode all remaining objects
    le = LabelEncoder()
    for col in ch.select_dtypes(include='object').columns:
        ch[col] = le.fit_transform(ch[col].astype(str))

    # Cell 17 – train/test split
    ch_train, ch_test = train_test_split(ch, test_size=0.2, random_state=42)

    def get_split(tr, te):
        return tr.iloc[:,:-1], tr.iloc[:,-1], te.iloc[:,:-1], te.iloc[:,-1]

    def train_eval(Xtr, ytr, Xte, yte):
        results = {}
        for name, mdl in [
            ('Logistic Regression', LogisticRegression(max_iter=1000)),
            ('Decision Tree',       DecisionTreeClassifier(random_state=42)),
            ('Random Forest',       RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)),
        ]:
            mdl.fit(Xtr, ytr)
            preds = mdl.predict(Xte)
            cm = confusion_matrix(yte, preds)
            cr = classification_report(yte, preds, output_dict=True)
            fi = None
            if hasattr(mdl,'feature_importances_'):
                fi = pd.DataFrame({'Feature':Xtr.columns,'Score':mdl.feature_importances_}).sort_values('Score',ascending=False).reset_index(drop=True)
            results[name] = dict(model=mdl, cm=cm, cr=cr, fi=fi)
        return results

    # ── Before oversampling ──
    Xtr, ytr, Xte, yte = get_split(ch_train, ch_test)
    before = train_eval(Xtr, ytr, Xte, yte)
    vc_before = ch_train['Churn'].value_counts().to_dict()

    # ── Oversampling (Cell 26) ──
    minority = ch_train[ch_train['Churn']==1]
    extra    = minority.iloc[:min(1000,len(minority))]
    ch_train_os = pd.concat([ch_train, extra], ignore_index=True)
    vc_after = ch_train_os['Churn'].value_counts().to_dict()

    Xtr2, ytr2, Xte2, yte2 = get_split(ch_train_os, ch_test)
    after = train_eval(Xtr2, ytr2, Xte2, yte2)

    feature_cols = Xtr.columns.tolist()

    return dict(
        nulls_before=nulls_before,
        obj_cols=obj_cols,
        tc_nulls=tc_nulls,
        ch_encoded=ch,
        ch_train=ch_train, ch_test=ch_test,
        feature_cols=feature_cols,
        vc_before=vc_before,
        before=before,
        vc_after=vc_after,
        after=after,
        rf_model=after['Random Forest']['model'],
    )


# ─────────────────────────────────────────────
#  Small helpers
# ─────────────────────────────────────────────
MODEL_COLORS = {
    'Logistic Regression': '#0d9488',
    'Decision Tree':       '#d97706',
    'Random Forest':       '#2563eb',
}

def metric_bar(label, val, color, max_v=1.0):
    pct = round(val/max_v*100, 1)
    return f"""<div class='mbar-wrap'>
  <div class='mbar-row'><span class='mbar-lbl'>{label}</span><span class='mbar-val'>{val:.4f}</span></div>
  <div class='mbar-bg'><div class='mbar-fill' style='width:{pct}%;background:{color}'></div></div>
</div>"""


def confusion_card(cm, color):
    tn,fp,fn,tp = cm.ravel()
    return f"""
<div class='cm-grid'>
  <div></div>
  <div class='cm-hd'>Predicted No</div>
  <div class='cm-hd'>Predicted Yes</div>
  <div class='cm-lbl'>Actual No</div>
  <div class='cm-cell cm-TN'><span class='big'>{tn}</span><span class='small'>True Neg</span></div>
  <div class='cm-cell cm-FP'><span class='big'>{fp}</span><span class='small'>False Pos</span></div>
  <div class='cm-lbl'>Actual Yes</div>
  <div class='cm-cell cm-FN'><span class='big'>{fn}</span><span class='small'>False Neg</span></div>
  <div class='cm-cell cm-TP'><span class='big'>{tp}</span><span class='small'>True Pos</span></div>
</div>"""


def model_block(name, result, extra_label=""):
    color = MODEL_COLORS[name]
    cr    = result['cr']
    cm    = result['cm']
    label = f"{name} {extra_label}".strip()

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"""
        <div class='model-wrap'>
          <div class='model-head'>
            <div class='model-dot' style='background:{color}'></div>
            <span class='model-name'>{label}</span>
            <span class='model-tag'>sklearn</span>
          </div>
          {metric_bar('Accuracy',              cr['accuracy'],       color)}
          {metric_bar('Precision  (Churn=1)',  cr['1']['precision'], color)}
          {metric_bar('Recall  (Churn=1)',     cr['1']['recall'],    color)}
          {metric_bar('F1-Score  (Churn=1)',   cr['1']['f1-score'],  color)}
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='model-wrap'><b style='font-size:.87rem'>Confusion Matrix</b>{confusion_card(cm, color)}</div>",
                    unsafe_allow_html=True)


def section(icon, label):
    st.markdown(f"<div class='sec'><div class='sec-icon'>{icon}</div>{label}</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Header + Upload
# ─────────────────────────────────────────────
col_h, col_u = st.columns([3,1])
with col_h:
    st.markdown("""
    <div class='nb-header'>
      <div>
        <div class='nb-badge'>🔬 Machine Learning · Classification</div>
        <div class='nb-title' style='margin-top:.5rem'>Customer <span>Churn</span> Prediction</div>
        <div class='nb-sub'>Logistic Regression · Decision Tree · Random Forest · Oversampling · Feature Importance</div>
      </div>
      <div style='text-align:right;font-size:.75rem;color:#a8a29e;line-height:2'>
        sklearn 1.x<br>pandas · numpy<br>Jupyter → Streamlit
      </div>
    </div>""", unsafe_allow_html=True)

with col_u:
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload churn.csv", type=["csv"], label_visibility="visible")
    run = st.button("▶  Run Pipeline", use_container_width=True)

# load data
if uploaded:
    raw = pd.read_csv(uploaded); st.session_state['raw'] = raw
elif 'raw' not in st.session_state:
    st.session_state['raw'] = make_demo()
raw = st.session_state['raw']

if run or 'pipe' not in st.session_state:
    with st.spinner("Running pipeline…"):
        st.session_state['pipe'] = run_pipeline(raw)

pipe = st.session_state['pipe']

# ─────────────────────────────────────────────
#  Tabs  (mirror notebook sections exactly)
# ─────────────────────────────────────────────
tabs = st.tabs([
    "📂  Data Loading",
    "🧹  Preprocessing",
    "📊  Before Oversampling",
    "⚖️  Oversampling",
    "📈  After Oversampling",
    "🌟  Feature Importance",
    "💾  Save Model",
])


# ════════════════════════════════════════════
# TAB 1 · Data Loading  (Cells 1–3)
# ════════════════════════════════════════════
with tabs[0]:
    churn_pct = (raw['Churn'].map({'Yes':1,'No':0}).mean()*100
                 if raw['Churn'].dtype==object else raw['Churn'].mean()*100)

    section("📋","Dataset Overview")
    st.markdown(f"""
    <div class='stat-row'>
      <div class='stat'><div class='stat-lbl'>Total Rows</div><div class='stat-val'>{len(raw):,}</div></div>
      <div class='stat blue'><div class='stat-lbl'>Columns</div><div class='stat-val'>{len(raw.columns)}</div></div>
      <div class='stat amber'><div class='stat-lbl'>Churn Rate</div><div class='stat-val'>{churn_pct:.1f}%</div></div>
      <div class='stat red'><div class='stat-lbl'>Object Cols</div><div class='stat-val'>{len(raw.select_dtypes('object').columns)}</div></div>
    </div>""", unsafe_allow_html=True)

    section("👁","ch.head(7)  —  First 7 Rows")
    st.dataframe(raw.head(7), use_container_width=True, height=260)

    section("📌","Problem Statement (Notebook Cells 2–3)")
    st.markdown("""
    <div class='step-list'>
      <div class='step-item'><div class='step-n'>1</div><div><div class='step-t'>Problem Statement</div><div class='step-d'>Predict the <b>Churn</b> column — whether a customer will leave (Yes/No → 1/0)</div></div></div>
      <div class='step-item'><div class='step-n'>2</div><div><div class='step-t'>Data Gathering</div><div class='step-d'>Data provided as <code>churn.csv</code></div></div></div>
      <div class='step-item'><div class='step-n'>3</div><div><div class='step-t'>Data Cleaning</div><div class='step-d'>Remove / impute nulls · fix TotalCharges blanks</div></div></div>
      <div class='step-item'><div class='step-n'>4</div><div><div class='step-t'>Sampling</div><div class='step-d'>80% train · 20% test — random split ensures no data leakage</div></div></div>
      <div class='step-item'><div class='step-n'>5</div><div><div class='step-t'>Fit Model</div><div class='step-d'>Build three classifiers on training data</div></div></div>
      <div class='step-item'><div class='step-n'>6</div><div><div class='step-t'>Predict & Evaluate</div><div class='step-d'>Confusion matrix + classification report on test data</div></div></div>
    </div>""", unsafe_allow_html=True)

    section("🏷","Categorical Columns  —  ch.select_dtypes(object)")
    st.dataframe(pipe['obj_cols'].head(5), use_container_width=True, height=210)


# ════════════════════════════════════════════
# TAB 2 · Preprocessing  (Cells 4–15)
# ════════════════════════════════════════════
with tabs[1]:
    section("🔍","Null Check  —  ch.isnull().sum()")
    if pipe['nulls_before'].empty:
        st.markdown("<div class='infobox'>✅ No explicit nulls found before cleaning.</div>", unsafe_allow_html=True)
    else:
        st.dataframe(pipe['nulls_before'].reset_index().rename(columns={'index':'Column',0:'Count'}), use_container_width=True)

    section("💡","Label Encoding Strategy  (Notebook Cell 5)")
    st.markdown("""
    <div class='card' style='font-size:.84rem;color:#57534e;line-height:1.75'>
      <b style='color:#1c1917'>Key rule:</b> ML models only accept numbers — all text columns must be encoded.<br>
      • <b>Ordinal data</b> → <code>replace()</code> with meaningful integers (e.g. Contract type).<br>
      • <b>Nominal data</b> → <code>LabelEncoder</code> (arbitrary integer per category).
    </div>""", unsafe_allow_html=True)

    section("🔢","Contract Encoding  —  Ordinal Replace  (Cell 8)")
    st.markdown("""
    <div class='codebox'>
      ch.Contract.<span class='kw'>replace</span>({<span class='str'>'Month-to-month'</span>: <b>0</b>, <span class='str'>'One year'</span>: <b>1</b>, <span class='str'>'Two year'</span>: <b>2</b>}, inplace=<span class='kw'>True</span>)
    </div>""", unsafe_allow_html=True)
    contract_map = pd.DataFrame({'Contract Type':['Month-to-month','One year','Two year'],'Encoded Value':[0,1,2],'Reasoning':['Shortest commitment','Medium commitment','Longest commitment']})
    st.dataframe(contract_map, use_container_width=True, hide_index=True)

    section("🗑","Drop customerID  (Cell 9)")
    st.markdown("""<div class='codebox'>ch = ch.<span class='kw'>drop</span>([<span class='str'>'customerID'</span>], axis=<b>1</b>)
<span class='cm'># non-predictive identifier — remove before modelling</span></div>""", unsafe_allow_html=True)

    section("🩹","Fix TotalCharges Blanks  (Cells 12–13)")
    st.markdown(f"""
    <div class='codebox'>ch.TotalCharges = ch.TotalCharges.<span class='kw'>replace</span>({{<span class='str'>''</span>: np.nan}})
<span class='cm'># converting blank spaces to NaN</span>
ch.TotalCharges.isnull().<span class='kw'>sum</span>()  <span class='cm'># → {pipe['tc_nulls']} missing values found</span></div>""",
    unsafe_allow_html=True)
    st.markdown(f"<div class='infobox'>⚠️ Found <b>{pipe['tc_nulls']}</b> blank TotalCharges entries — imputed with column median.</div>", unsafe_allow_html=True)

    section("🏷","Label Encode All Object Columns  (Cell 15)")
    st.markdown("""<div class='codebox'>le = <span class='kw'>LabelEncoder</span>()
ch[ch.select_dtypes(<span class='str'>'object'</span>).columns] = ch[ch.select_dtypes(<span class='str'>'object'</span>).columns].<span class='kw'>apply</span>(le.fit_transform)</div>""", unsafe_allow_html=True)
    st.dataframe(pipe['ch_encoded'].head(7), use_container_width=True, height=250)

    section("✂️","Train / Test Split  (Cell 17)")
    n_tr = len(pipe['ch_train']); n_te = len(pipe['ch_test'])
    st.markdown(f"""
    <div class='codebox'>ch_train, ch_test = <span class='kw'>train_test_split</span>(ch, test_size=<b>0.2</b>)
<span class='cm'># {n_tr} train rows  ·  {n_te} test rows</span></div>""", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown(f"""<div class='stat'><div class='stat-lbl'>Train Set</div><div class='stat-val'>{n_tr:,}</div><div class='stat-note'>80% of data</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='stat blue'><div class='stat-lbl'>Test Set</div><div class='stat-val'>{n_te:,}</div><div class='stat-note'>20% of data</div></div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════
# TAB 3 · Before Oversampling  (Cells 19–21)
# ════════════════════════════════════════════
with tabs[2]:
    section("📊","Class Distribution  (Before Oversampling)")
    vc = pipe['vc_before']
    total = sum(vc.values())
    for cls, cnt in sorted(vc.items()):
        lbl = "Churn (Yes)" if cls==1 else "No Churn (No)"
        color = "#dc2626" if cls==1 else "#0d9488"
        pct = cnt/total
        st.markdown(f"""<div class='os-bar-wrap'>
          <span class='os-bar-lbl'>{lbl}</span>
          <div class='os-bar-bg'><div class='os-bar-fill' style='width:{pct*100:.1f}%;background:{color}'></div></div>
          <span class='os-bar-cnt'>{cnt:,}</span>
        </div>""", unsafe_allow_html=True)

    section("📈","Model Results  (Before Oversampling)")
    for name, result in pipe['before'].items():
        model_block(name, result, "— Before Oversampling")


# ════════════════════════════════════════════
# TAB 4 · Oversampling  (Cells 22–27)
# ════════════════════════════════════════════
with tabs[3]:
    section("⚖️","What is Oversampling?")
    st.markdown("""
    <div class='card' style='font-size:.84rem;color:#57534e;line-height:1.8'>
      <b style='color:#1c1917'>Problem:</b> The dataset is imbalanced — far fewer customers actually churned.<br>
      <b style='color:#1c1917'>Solution:</b> Duplicate minority class (Churn=1) rows to balance the training set.<br>
      This prevents the model from being biased toward predicting "No Churn" every time.
    </div>""", unsafe_allow_html=True)

    section("🔬","Notebook Cells 23–26")
    st.markdown("""
    <div class='codebox'><span class='cm'># Check class counts</span>
ch_train.Churn.<span class='kw'>value_counts</span>()

<span class='cm'># Isolate minority class</span>
df = ch_train[ch_train.Churn == <b>1</b>]

<span class='cm'># Add 1000 duplicated minority rows</span>
ch_train = pd.<span class='kw'>concat</span>([ch_train, df.iloc[0:<b>1000</b>]])
ch_train.Churn.<span class='kw'>value_counts</span>()</div>""", unsafe_allow_html=True)

    section("📊","Before vs After Oversampling")
    vc_b = pipe['vc_before']; vc_a = pipe['vc_after']
    total_a = sum(vc_a.values())
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("**Before**")
        total_b = sum(vc_b.values())
        for cls,cnt in sorted(vc_b.items()):
            lbl = "Churn=1" if cls==1 else "Churn=0"
            color = "#dc2626" if cls==1 else "#0d9488"
            st.markdown(f"""<div class='os-bar-wrap'><span class='os-bar-lbl'>{lbl}</span>
              <div class='os-bar-bg'><div class='os-bar-fill' style='width:{cnt/total_b*100:.1f}%;background:{color}'></div></div>
              <span class='os-bar-cnt'>{cnt:,}</span></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("**After Oversampling**")
        for cls,cnt in sorted(vc_a.items()):
            lbl = "Churn=1" if cls==1 else "Churn=0"
            color = "#dc2626" if cls==1 else "#0d9488"
            st.markdown(f"""<div class='os-bar-wrap'><span class='os-bar-lbl'>{lbl}</span>
              <div class='os-bar-bg'><div class='os-bar-fill' style='width:{cnt/total_a*100:.1f}%;background:{color}'></div></div>
              <span class='os-bar-cnt'>{cnt:,}</span></div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════
# TAB 5 · After Oversampling  (Cells 28–30)
# ════════════════════════════════════════════
with tabs[4]:
    section("📈","Model Results  (After Oversampling)")
    for name, result in pipe['after'].items():
        model_block(name, result, "— After Oversampling")

    section("🏆","Comparison: Before vs After Oversampling")
    rows = []
    for nm in pipe['before']:
        b = pipe['before'][nm]['cr']
        a = pipe['after'][nm]['cr']
        rows.append({
            'Model': nm,
            'F1 Before': round(b['1']['f1-score'],4),
            'F1 After':  round(a['1']['f1-score'],4),
            'Recall Before': round(b['1']['recall'],4),
            'Recall After':  round(a['1']['recall'],4),
        })
    cmp = pd.DataFrame(rows).set_index('Model')
    st.dataframe(cmp.style.highlight_max(axis=0,color='#ccfbf1'), use_container_width=True)


# ════════════════════════════════════════════
# TAB 6 · Feature Importance  (Cell 31)
# ════════════════════════════════════════════
with tabs[5]:
    section("🌟","Feature Importances — Random Forest  (Cell 31)")
    st.markdown("""<div class='codebox'>feat_imp = pd.<span class='kw'>DataFrame</span>({<span class='str'>'Features'</span>: ch_train_x.columns, <span class='str'>'Score'</span>: lr.feature_importances_})
feat_imp = feat_imp.<span class='kw'>sort_values</span>(<span class='str'>'Score'</span>, ascending=<span class='kw'>False</span>)</div>""", unsafe_allow_html=True)

    fi = pipe['after']['Random Forest']['fi']
    if fi is not None:
        max_score = fi['Score'].max()
        for i, row in fi.iterrows():
            rank = i+1
            pct  = row['Score']/max_score*100
            st.markdown(f"""<div class='fi-row'>
              <span class='fi-rank'>#{rank:02d}</span>
              <span class='fi-name'>{row['Feature']}</span>
              <div class='fi-bg'><div class='fi-fill' style='width:{pct:.1f}%'></div></div>
              <span class='fi-score'>{row['Score']:.4f}</span>
            </div>""", unsafe_allow_html=True)

        section("📋","Full Feature Importance Table")
        st.dataframe(fi.rename(columns={'Feature':'Features','Score':'Score'}),
                     use_container_width=True, hide_index=True)


# ════════════════════════════════════════════
# TAB 7 · Save Model  (Cell 32)
# ════════════════════════════════════════════
with tabs[6]:
    section("💾","Save Trained Model  —  pickle.dump()  (Cell 32)")
    st.markdown("""<div class='codebox'><span class='kw'>import</span> pickle

<span class='cm'># Save the trained Random Forest model</span>
pickle.<span class='kw'>dump</span>(lr, <span class='kw'>open</span>(<span class='str'>"credit_model.pkl"</span>, <span class='str'>"wb"</span>))</div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class='step-list' style='margin-bottom:1.2rem'>
      <div class='step-item'><div class='step-n'>1</div><div><div class='step-t'>What pickle does</div><div class='step-d'>Serializes the Python object (model) into a binary file that can be reloaded later</div></div></div>
      <div class='step-item'><div class='step-n'>2</div><div><div class='step-t'>Why save the model</div><div class='step-d'>Allows deployment — load once, predict many times without retraining</div></div></div>
      <div class='step-item'><div class='step-n'>3</div><div><div class='step-t'>Loading back</div><div class='step-d'><code>model = pickle.load(open("credit_model.pkl", "rb"))</code></div></div></div>
    </div>""", unsafe_allow_html=True)

    buf = io.BytesIO()
    pickle.dump(pipe['rf_model'], buf)
    buf.seek(0)

    st.markdown("<div class='infobox'>✅ Random Forest (After Oversampling) is used as the final model — highest recall on churn class.</div>", unsafe_allow_html=True)
    st.download_button(
        label="⬇️  Download  credit_model.pkl",
        data=buf,
        file_name="credit_model.pkl",
        mime="application/octet-stream",
    )