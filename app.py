import sys
import asyncio

# Prevent WinError 10054 on Windows socket reset
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

pipeline = joblib.load('models/best_pipeline.pkl')

FAVICON = """data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="%23c084fc"><path d="M12 2a9 9 0 0 0-9 9c0 3.87 2.45 7.17 5.92 8.42.45.08.62-.2.62-.44v-1.54c-2.42.53-2.93-1.17-2.93-1.17-.4-.99-.97-1.26-.97-1.26-.79-.54.06-.53.06-.53.87.06 1.33.9 1.33.9.77 1.33 2.03.95 2.53.72.08-.56.3-.95.55-1.17-1.93-.22-3.96-.97-3.96-4.31 0-.95.34-1.73.9-2.34-.09-.22-.39-1.11.09-2.31 0 0 .73-.23 2.4 1.12a8.38 8.38 0 0 1 4.38 0c1.67-1.35 2.4-1.12 2.4-1.12.48 1.2.18 2.09.09 2.31.56.61.9 1.39.9 2.34 0 3.35-2.03 4.09-3.97 4.31.31.27.59.8.59 1.62v2.4c0 .24.16.53.62.44A9.003 9.003 0 0 0 21 11a9 9 0 0 0-9-9z"/></svg>"""

st.set_page_config(
    page_title="PLACEMENT // CORE AI",
    page_icon=FAVICON,
    layout="wide"
)

def svg_icon(path_d, color="#c084fc", size=16, viewBox="0 0 24 24"):
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="{viewBox}" width="{size}" height="{size}" fill="{color}" style="vertical-align: -2px; display: inline-block;"><path d="{path_d}"/></svg>"""

# SVG Icon Paths
ICO_CHIP = "M6 2v2H4c-.55 0-1 .45-1 1v2H1v2h2v2H1v2h2v2H1v2h2v2c0 .55.45 1 1 1h2v2h2v-2h2v2h2v-2h2v2h2v-2h2c.55 0 1-.45 1-1v-2h2v-2h-2v-2h2v-2h-2v-2h2V7h-2V5c0-.55-.45-1-1-1h-2V2h-2v2h-2V2h-2v2H8V2H6zm2 4h8v8H8V6zm2 2v4h4V8h-4z"
ICO_SEARCH = "M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"
ICO_ID = "M20 4H4c-1.11 0-1.99.89-1.99 2L2 18c0 1.11.89 2 2 2h16c1.11 0 2-.89 2-2V6c0-1.11-.89-2-2-2zm-9 3c1.66 0 3 1.34 3 3s-1.34 3-3 3-3-1.34-3-3 1.34-3 3-3zm6 10H5v-.5c0-1.66 3.33-2.5 5-2.5s5 .84 5 2.5v.5zm3-4h-5v-1h5v1zm0-2h-5v-1h5v1zm0-2h-5V9h5v1z"
ICO_LAYERS = "M11.99 18.54l-7.37-5.73L3 14.07l9 7 9-7-1.63-1.27-7.38 5.74zM12 16l7.36-5.73L21 9.07l-9-7-9 7 1.63 1.2L12 16z"
ICO_BRANCH = "M6 2a3 3 0 0 0-3 3c0 1.31.84 2.42 2 2.83V16.17c-1.16.41-2 1.52-2 2.83a3 3 0 1 0 5-2.24V14a3 3 0 0 1 3-3h4.17c.41 1.16 1.52 2 2.83 2a3 3 0 1 0-3-3H14a5 5 0 0 0-5 5v.17A3.001 3.001 0 0 0 6 2zm0 2a1 1 0 1 1 0 2 1 1 0 0 1 0-2zm12 7a1 1 0 1 1 0 2 1 1 0 0 1 0-2zm-12 7a1 1 0 1 1 0 2 1 1 0 0 1 0-2z"
ICO_CHECK = "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"
ICO_BAN = "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.42 0-8-3.58-8-8 0-1.85.63-3.55 1.69-4.9L16.9 18.31C15.55 19.37 13.85 20 12 20zm6.31-3.1L7.1 5.69C8.45 4.63 10.15 4 12 4c4.42 0 8 3.58 8 8 0 1.85-.63 3.55-1.69 4.9z"
ICO_BRIEFCASE = "M20 6h-4V4c0-1.11-.89-2-2-2h-4c-1.11 0-2 .89-2 2v2H4c-1.11 0-1.99.89-1.99 2L2 19c0 1.11.89 2 2 2h16c1.11 0 2-.89 2-2V8c0-1.11-.89-2-2-2zm-6 0h-4V4h4v2z"
ICO_REPORT = "M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"
ICO_ALERT = "M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"

@st.cache_data
def load_student_data():
    for path in ['data/student_placement_data.csv', 'data/student_placement_data_v2.csv', 'student_placement_data.csv']:
        if os.path.exists(path):
            return pd.read_csv(path)
    raise FileNotFoundError("Could not locate student placement dataset.")

df_students = load_student_data()

# ----------------- MONOSPACE ROBOTIC + PURPLE THEME -----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700;800&family=Share+Tech+Mono&display=swap');

*, html, body, [class*="css"], [class*="st-"], .stMarkdown, .stText, p, span, label, input, button, select, div {
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
}

h1, h2, h3, h4, h5, h6, .brand-title {
    font-family: 'Share Tech Mono', monospace !important;
    letter-spacing: 1.2px;
}

.stApp {
    background-color: #06060c !important;
    background-image: 
        radial-gradient(at 10% 10%, rgba(147, 51, 234, 0.22) 0px, transparent 45%),
        radial-gradient(at 90% 90%, rgba(192, 132, 252, 0.15) 0px, transparent 45%),
        radial-gradient(at 50% 50%, rgba(79, 70, 229, 0.08) 0px, transparent 65%) !important;
    background-attachment: fixed !important;
    color: #f1f5f9 !important;
}

.purple-hero {
    background: linear-gradient(135deg, rgba(30, 20, 60, 0.8) 0%, rgba(88, 28, 135, 0.45) 50%, rgba(12, 8, 24, 0.85) 100%);
    border-left: 5px solid #a855f7;
    border-top: 1px solid rgba(168, 85, 247, 0.35);
    border-right: 1px solid rgba(168, 85, 247, 0.35);
    border-bottom: 1px solid rgba(168, 85, 247, 0.35);
    border-radius: 0 14px 14px 0;
    padding: 24px 30px;
    margin-bottom: 24px;
    box-shadow: 0 10px 35px -5px rgba(139, 92, 246, 0.28);
    backdrop-filter: blur(16px);
}

.hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(168, 85, 247, 0.2);
    border: 1px solid rgba(192, 132, 252, 0.45);
    color: #d8b4fe;
    padding: 4px 14px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 8px;
}

.brand-title {
    font-size: 2.1rem;
    margin: 0;
    color: #f5f3ff !important;
    text-shadow: 0 0 16px rgba(192, 132, 252, 0.5);
}

.section-banner {
    display: flex;
    align-items: center;
    gap: 12px;
    background: rgba(147, 51, 234, 0.12);
    border: 1px solid rgba(168, 85, 247, 0.25);
    border-left: 4px solid #c084fc;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    margin: 16px 0 14px 0;
}

.section-title {
    font-size: 1.02rem;
    font-weight: 700;
    color: #e9d5ff;
    margin: 0;
    letter-spacing: 0.8px;
}

div[data-testid="stForm"] {
    background: rgba(15, 10, 28, 0.65) !important;
    border: 1px solid rgba(147, 51, 234, 0.28) !important;
    border-radius: 14px !important;
    padding: 24px !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6) !important;
    backdrop-filter: blur(14px) !important;
}

div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #7c3aed 0%, #9333ea 50%, #c026d3 100%) !important;
    color: #ffffff !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 14px 28px !important;
    width: 100% !important;
    box-shadow: 0 0 25px rgba(147, 51, 234, 0.45) !important;
    transition: all 0.3s ease !important;
}
div[data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 35px rgba(168, 85, 247, 0.8) !important;
}

.card-placed {
    background: linear-gradient(135deg, rgba(88, 28, 135, 0.4) 0%, rgba(15, 10, 28, 0.85) 100%);
    border-left: 5px solid #a855f7;
    border-top: 1px solid rgba(192, 132, 252, 0.4);
    border-right: 1px solid rgba(192, 132, 252, 0.4);
    border-bottom: 1px solid rgba(192, 132, 252, 0.4);
    border-radius: 0 10px 10px 0;
    padding: 22px;
}

.card-unplaced {
    background: linear-gradient(135deg, rgba(76, 5, 25, 0.4) 0%, rgba(15, 10, 28, 0.85) 100%);
    border-left: 5px solid #ef4444;
    border-top: 1px solid rgba(244, 63, 94, 0.35);
    border-right: 1px solid rgba(244, 63, 94, 0.35);
    border-bottom: 1px solid rgba(244, 63, 94, 0.35);
    border-radius: 0 10px 10px 0;
    padding: 22px;
}

.metric-tier-card {
    background: rgba(20, 14, 38, 0.85);
    border: 1px solid rgba(147, 51, 234, 0.35);
    border-radius: 10px;
    padding: 22px;
}

.pill-badge {
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 8px;
}
.badge-lx { background: rgba(168, 85, 247, 0.18); border: 1px solid rgba(168, 85, 247, 0.4); color: #d8b4fe; }
.badge-ax { background: rgba(129, 140, 248, 0.18); border: 1px solid rgba(129, 140, 248, 0.4); color: #a5b4fc; }
.badge-cx { background: rgba(192, 132, 252, 0.22); border: 1px solid rgba(192, 132, 252, 0.5); color: #e9d5ff; }
.badge-px { background: rgba(232, 121, 249, 0.18); border: 1px solid rgba(232, 121, 249, 0.4); color: #f0abfc; }
.badge-sx { background: rgba(244, 114, 182, 0.18); border: 1px solid rgba(244, 114, 182, 0.4); color: #f472b6; }
</style>
""", unsafe_allow_html=True)

# ----------------- HERO HEADER -----------------
st.markdown(f"""
<div class="purple-hero">
    <div class="hero-pill">
        {svg_icon(ICO_CHIP, '#d8b4fe', 14)}
        <span>SYS_ID: 1CR23AI119 // PLACEMENT CORE AI</span>
    </div>
    <h1 class="brand-title">PLACEMENT PREDICTION & READINESS MATRIX</h1>
    <p style="color: #c4b5fd; font-size: 0.85rem; margin: 6px 0 0 0;">
        {svg_icon(ICO_BRANCH, '#c084fc', 14)} ENGINE: CALIBRATED RANDOM FOREST // RUBRIC: 5-TIER MODULAR MATRIX
    </p>
</div>
""", unsafe_allow_html=True)

# ----------------- CANDIDATE LOOKUP -----------------
st.markdown(f"""
<div class="section-banner">
    {svg_icon(ICO_SEARCH, '#c084fc', 18)}
    <span class="section-title">Candidate Directory & Auto-Population</span>
</div>
""", unsafe_allow_html=True)

has_meta = ('usn' in df_students.columns and 'student_name' in df_students.columns)

if has_meta:
    options = ['[+] Custom Candidate Entry'] + [
        f"{row.usn} - {row.student_name} ({row.branch} | CGPA: {row.cgpa})"
        for _, row in df_students.head(300).iterrows()
    ]
else:
    options = ['[+] Custom Candidate Entry'] + [
        f"ID #{row.student_id:04d} ({row.branch} | CGPA: {row.cgpa})"
        for _, row in df_students.head(300).iterrows()
    ]

selected = st.selectbox("Candidate Search:", options, label_visibility="collapsed")
is_custom = (selected == '[+] Custom Candidate Entry')

if is_custom:
    d_name, d_usn = "", ""  # Clean ghost placeholder
    d_gender, d_age, d_degree, d_branch = "Male", 21, "BTech", "CS"
    d_tenth, d_twelfth = 82.5, 80.0
    d_cgpa, d_back, d_int, d_cert, d_proj = 8.10, 0, 1, 2, 2
    d_coding, d_comm, d_apt = 7, 7, 75
    d_lx, d_ax, d_cx, d_px, d_sx = 3, 3, 3, 3.5, 3
else:
    if has_meta:
        sel_usn = selected.split(' - ')[0]
        row = df_students[df_students['usn'] == sel_usn].iloc[0]
        d_name, d_usn = row['student_name'], row['usn']
    else:
        sel_id = int(selected.split(' (')[0].replace('ID #', ''))
        row = df_students[df_students['student_id'] == sel_id].iloc[0]
        d_name, d_usn = f"Student #{sel_id}", f"1CR23CS{sel_id:04d}"
        
    d_gender, d_age, d_degree, d_branch = row['gender'], int(row['age']), row['degree'], row['branch']
    d_cgpa, d_back = float(row['cgpa']), int(row['backlogs'])
    # Correlate historical 10th/12th realistically with college CGPA
    d_tenth = round(min(98.0, max(52.0, d_cgpa * 9.5 + 4.0)), 1)
    d_twelfth = round(min(98.0, max(50.0, d_cgpa * 9.2 + 2.0)), 1)
    d_int, d_cert, d_proj = int(row['internships']), int(row['certifications']), int(row['projects'])
    d_coding, d_comm, d_apt = int(row['coding_skills']), int(row['communication_skills']), int(row['aptitude_score'])
    d_lx = int(row['Lx_Level_Reached'])
    d_ax = int(row['Ax_Level_Reached'])
    d_cx = int(row['Cx_Level_Reached'])
    d_px = float(row['Px_Level_Reached'])
    d_sx = int(row['Sx_Level_Reached'])

# ----------------- INPUT FORM -----------------
with st.form("student_form"):
    st.markdown(f"""
    <div class="section-banner">
        {svg_icon(ICO_ID, '#c084fc', 18)}
        <span class="section-title">Academic & Identity Credentials</span>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        s_name = st.text_input("Full Name", value=d_name, placeholder="Enter student name...")
        s_usn = st.text_input("University USN", value=d_usn, placeholder="e.g. 1CR23CS0142")
        gender = st.selectbox("Gender", ["Male", "Female"], index=0 if d_gender == "Male" else 1)
        age = st.number_input("Age", 18, 30, value=d_age)
        
    with c2:
        degree = st.selectbox("Degree Program", ["BTech", "BE", "BCA", "BSc"], 
                              index=["BTech", "BE", "BCA", "BSc"].index(d_degree) if d_degree in ["BTech", "BE", "BCA", "BSc"] else 0)
        branch = st.selectbox("Discipline / Branch", ["CS", "IT", "AI", "DS", "Electrical", "Mechanical"],
                              index=["CS", "IT", "AI", "DS", "Electrical", "Mechanical"].index(d_branch) if d_branch in ["CS", "IT", "AI", "DS", "Electrical", "Mechanical"] else 0)
        tenth_pct = st.number_input("10th / Secondary Board (%)", 35.0, 100.0, value=float(d_tenth), step=0.5, format="%.1f")
        twelfth_pct = st.number_input("12th / PUC / Diploma (%)", 35.0, 100.0, value=float(d_twelfth), step=0.5, format="%.1f")
        
    with c3:
        cgpa = st.number_input("College Cumulative CGPA", 0.0, 10.0, value=d_cgpa, step=0.05, format="%.2f")
        backlogs = st.number_input("Active Backlogs (0 Mandatory)", 0, 10, value=d_back)
        internships = st.number_input("Internships Completed", 0, 10, value=d_int)
        certifications = st.number_input("Certifications Earned", 0, 15, value=d_cert)

    c4, c5, c6 = st.columns(3)
    with c4:
        projects = st.number_input("Capstone Projects", 0, 15, value=d_proj)
    with c5:
        coding_skills = st.slider("Coding Fluency (1-10)", 1, 10, value=d_coding)
    with c6:
        communication_skills = st.slider("Communication Index (1-10)", 1, 10, value=d_comm)
        aptitude_score = st.slider("Aptitude Benchmark (40-99)", 40, 99, value=d_apt)

    st.markdown(f"""
    <div class="section-banner">
        {svg_icon(ICO_LAYERS, '#c084fc', 18)}
        <span class="section-title">Department Modular Cutoff Clearances</span>
    </div>
    """, unsafe_allow_html=True)
    
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown('<span class="pill-badge badge-lx">✦ Language (Lx)</span>', unsafe_allow_html=True)
        lx = st.selectbox("Level Cleared", [0, 1, 2, 3, 4], index=[0, 1, 2, 3, 4].index(int(d_lx)), key="lx_box")
    with m2:
        st.markdown('<span class="pill-badge badge-ax">✦ Aptitude (Ax)</span>', unsafe_allow_html=True)
        ax = st.selectbox("Level Cleared", [0, 1, 2, 3, 4], index=[0, 1, 2, 3, 4].index(int(d_ax)), key="ax_box")
    with m3:
        st.markdown('<span class="pill-badge badge-cx">✦ Core Test (Cx)</span>', unsafe_allow_html=True)
        cx = st.selectbox("Level Cleared", [0, 2, 3, 4, 5], index=[0, 2, 3, 4, 5].index(int(d_cx)) if int(d_cx) in [0, 2, 3, 4, 5] else 2, key="cx_box")
    with m4:
        st.markdown('<span class="pill-badge badge-px">✦ Prog (Px)</span>', unsafe_allow_html=True)
        px = st.selectbox("Level Cleared", [0.0, 1.0, 2.0, 3.0, 3.5, 4.0, 5.0], index=[0.0, 1.0, 2.0, 3.0, 3.5, 4.0, 5.0].index(float(d_px)), key="px_box")
    with m5:
        st.markdown('<span class="pill-badge badge-sx">✦ Softskills (Sx)</span>', unsafe_allow_html=True)
        sx = st.selectbox("Level Cleared", [0, 1, 2, 3, 4], index=[0, 1, 2, 3, 4].index(int(d_sx)), key="sx_box")

    st.write("")
    submitted = st.form_submit_button(">> EXECUTE PLACEMENT ANALYSIS")

# ----------------- DIAGNOSTIC INFERENCE & REALISTIC ELIGIBILITY -----------------
if submitted:
    final_name = s_name.strip() if s_name.strip() else ("Candidate" if is_custom else d_name)
    final_usn = s_usn.strip() if s_usn.strip() else ("1CR23CS9999" if is_custom else d_usn)
    
    student_data = pd.DataFrame([{
        'gender': gender,
        'age': age,
        'degree': degree,
        'branch': branch,
        'cgpa': cgpa,
        'backlogs': backlogs,
        'internships': internships,
        'certifications': certifications,
        'coding_skills': coding_skills,
        'communication_skills': communication_skills,
        'aptitude_score': aptitude_score,
        'projects': projects,
        'Lx_Level_Reached': lx,
        'Ax_Level_Reached': ax,
        'Cx_Level_Reached': cx,
        'Px_Level_Reached': px,
        'Sx_Level_Reached': sx
    }])
    
    pred = pipeline.predict(student_data)[0]
    raw_prob = pipeline.predict_proba(student_data)[0][1] * 100
    overall = min(lx, ax, cx, px, sx)

    # 1. HARD CORPORATE FILTER 1: Active Backlogs
    has_backlogs = (backlogs > 0)

    # 2. HARD CORPORATE FILTER 2: 10th / 12th Board Cutoff (< 60.0% is disqualification for ~90% companies)
    board_below_60 = (tenth_pct < 60.0) or (twelfth_pct < 60.0)
    cgpa_below_6 = (cgpa < 6.0)

    # Realistic CGPA impact calculation
    # If CGPA is 7.0-7.5, probability drops significantly compared to an 8.5+ candidate
    cgpa_modifier = (cgpa - 7.8) * 8.0  # +8% per CGPA point above 7.8, or drops heavily below 7.8

    if has_backlogs:
        placed_prob = 0.0
        status_placed = False
        rejection_reason = "ACTIVE_BACKLOGS"
    elif board_below_60:
        # If below 60% in 10th or 12th, chances collapse down to 4-12% regardless of levels
        placed_prob = np.clip(12.0 - (60.0 - min(tenth_pct, twelfth_pct)) * 0.8, 2.0, 14.0)
        status_placed = False
        rejection_reason = "BOARD_MARKS_BELOW_60"
    elif cgpa_below_6:
        # College CGPA below 6.0 (First class cutoff)
        placed_prob = np.clip(10.0 + (cgpa - 5.0) * 8.0, 3.0, 18.0)
        status_placed = False
        rejection_reason = "CGPA_BELOW_FIRST_CLASS"
    elif pred == 1:
        # Base probability with CGPA drop curve
        placed_prob = np.clip(raw_prob + cgpa_modifier + (min(tenth_pct, twelfth_pct) - 75.0) * 0.2, 52.0, 98.5)
        
        # If CGPA is between 6.0 and 7.2, cap maximum probability to reflect real-world market shrinkage
        if cgpa < 7.0:
            placed_prob = min(placed_prob, 64.0)  # Heavy drop at 6.x CGPA
        elif cgpa < 7.5:
            placed_prob = min(placed_prob, 76.0)  # Moderate drop at 7.0-7.4 CGPA
            
        status_placed = True
        rejection_reason = None
    else:
        # Failed assessment tier rubrics
        placed_prob = np.clip(raw_prob * 0.4 + (overall / 3.0) * 10.0 + (cgpa / 10.0) * 8.0, 3.0, 42.0)
        status_placed = False
        rejection_reason = "TIER_CUTOFF_FAILED"

    st.markdown(f"""
    <div class="section-banner">
        {svg_icon(ICO_REPORT, '#c084fc', 18)}
        <span class="section-title">Diagnostic Placement Report</span>
    </div>
    """, unsafe_allow_html=True)
    
    col_res, col_metrics = st.columns([1.4, 1.0])
    
    with col_res:
        if rejection_reason == "ACTIVE_BACKLOGS":
            st.markdown(f"""
            <div class="card-unplaced" style="border-left: 5px solid #ef4444;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h2 style="color: #f87171; margin: 0; font-weight:800; font-size:1.35rem;">
                        {svg_icon(ICO_BAN, '#ef4444', 22)} DISQUALIFIED: ACTIVE BACKLOGS
                    </h2>
                    <span style="background: rgba(239, 68, 68, 0.25); border: 1px solid rgba(248, 113, 113, 0.5); color: #fca5a5; padding: 4px 14px; border-radius: 4px; font-weight:700; font-size:0.85rem;">
                        0.0% ELIGIBILITY
                    </span>
                </div>
                <p style="color: #fca5a5; margin: 10px 0 12px 0; font-weight:600;">
                    {final_name} [{final_usn}] // ACTIVE BACKLOGS: {backlogs}
                </p>
                <div style="background: rgba(10, 5, 20, 0.85); padding: 14px 18px; border-radius: 6px; border: 1px solid rgba(239, 68, 68, 0.35);">
                    <div style="font-size:0.8rem; text-transform:uppercase; color:#fda4af;">Mandatory Placement Drive Policy</div>
                    <div style="font-size: 0.9rem; color: #fecdd3; margin-top: 4px; line-height: 1.6;">
                        Even if assessment tiers are cleared (Level {overall}), campus recruitment portals automatically filter out candidates with active backlogs. Clear backlog to unlock drive registration.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif rejection_reason == "BOARD_MARKS_BELOW_60":
            failed_board = "10th Grade" if tenth_pct < 60.0 else "12th Grade"
            st.markdown(f"""
            <div class="card-unplaced" style="border-left: 5px solid #f59e0b;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h2 style="color: #fbbf24; margin: 0; font-weight:800; font-size:1.35rem;">
                        {svg_icon(ICO_ALERT, '#fbbf24', 22)} SEVERE RISK: BOARD MARKS &lt; 60%
                    </h2>
                    <span style="background: rgba(245, 158, 11, 0.25); border: 1px solid rgba(251, 191, 36, 0.5); color: #fde68a; padding: 4px 14px; border-radius: 4px; font-weight:700; font-size:0.85rem;">
                        {placed_prob:.1f}% CRITICAL RISK
                    </span>
                </div>
                <p style="color: #fde68a; margin: 10px 0 12px 0; font-weight:600;">
                    {final_name} [{final_usn}] // 10th: {tenth_pct}% | 12th: {twelfth_pct}%
                </p>
                <div style="background: rgba(10, 5, 20, 0.85); padding: 14px 18px; border-radius: 6px; border: 1px solid rgba(245, 158, 11, 0.35);">
                    <div style="font-size:0.8rem; text-transform:uppercase; color:#fbbf24;">Corporate 60% First Class Barrier</div>
                    <div style="font-size: 0.9rem; color: #fef3c7; margin-top: 4px; line-height: 1.6;">
                        Over 90% of campus hiring companies enforce a <b>strict 60% aggregate cutoff in 10th and 12th</b>. Because your score in {failed_board} is below 60%, corporate ATS filters will reject the candidate before the assessment round.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif status_placed:
            # Package allocation with realistic CGPA scaling
            if overall == 3.0:
                pkg, tier_title = "4.0 - 5.0 LPA", "Mass Recruiter / IT Services"
            elif overall == 3.5:
                pkg, tier_title = "5.0 - 8.0 LPA", "Digital Specialist / Consulting"
            elif overall == 4.0:
                pkg = "8.0 - 12.0 LPA" if cgpa < 7.8 else "8.0 - 14.0 LPA"
                tier_title = "Mid-tier Product / FinTech"
            else:
                pkg = "12.0 - 16.0 LPA" if cgpa < 8.0 else "14.0 - 20.0 LPA"
                tier_title = "Tier-1 Tech MNC / AI Labs"

            # Dynamic insight note on CGPA impact
            if cgpa >= 8.5:
                cgpa_note = "High CGPA (> 8.5) unlocks 100% of visiting campus companies."
            elif cgpa >= 7.5:
                cgpa_note = "CGPA between 7.5 and 8.5 satisfies cutoffs for ~85% of tier-1 & tier-2 firms."
            elif cgpa >= 7.0:
                cgpa_note = "CGPA in 7.0-7.4 band reduces probability. Filtered out of high-paying companies requiring 7.5+ cutoff."
            else:
                cgpa_note = "CGPA in 6.0-6.9 band is near the 60% boundary. Heavy competition in mass drives."

            st.markdown(f"""
            <div class="card-placed">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h2 style="color: #c084fc; margin: 0; font-weight:800; font-size:1.35rem;">
                        {svg_icon(ICO_CHECK, '#10b981', 22)} STATUS: PLACED
                    </h2>
                    <span style="background: rgba(168, 85, 247, 0.25); border: 1px solid rgba(192, 132, 252, 0.5); color: #e9d5ff; padding: 4px 14px; border-radius: 4px; font-weight:700; font-size:0.85rem;">
                        {placed_prob:.1f}% PROBABILITY
                    </span>
                </div>
                <p style="color: #e2e8f0; margin: 10px 0 16px 0;">
                    Candidate <b>{final_name}</b> [{final_usn}] satisfies all test tiers, 0-backlog policy, and 60%+ board cutoffs.
                </p>
                <div style="background: rgba(10, 5, 20, 0.8); padding: 14px 18px; border-radius: 6px; border: 1px solid rgba(168, 85, 247, 0.3);">
                    <div style="font-size:0.8rem; text-transform:uppercase; color:#c4b5fd;">
                        {svg_icon(ICO_BRIEFCASE, '#c4b5fd', 15)} Eligible Compensation Band
                    </div>
                    <div style="font-size: 1.6rem; font-weight: 800; color: #e9d5ff; margin: 2px 0;">{pkg}</div>
                    <div style="font-size: 0.85rem; color: #a5b4fc;">Target Tier: <b>{tier_title}</b></div>
                    <div style="margin-top: 8px; font-size: 0.8rem; color: #94a3b8; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 6px;">
                        <i>💡 {cgpa_note}</i>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="card-unplaced">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h2 style="color: #fb7185; margin: 0; font-weight:800; font-size:1.35rem;">
                        {svg_icon(ICO_BAN, '#fb7185', 22)} STATUS: NOT PLACED
                    </h2>
                    <span style="background: rgba(244, 63, 94, 0.2); border: 1px solid rgba(244, 63, 94, 0.4); color: #fda4af; padding: 4px 14px; border-radius: 4px; font-weight:700; font-size:0.85rem;">
                        {100 - placed_prob:.1f}% RISK
                    </span>
                </div>
                <p style="color: #e2e8f0; margin: 10px 0 0 0;">
                    Candidate <b>{final_name}</b> [{final_usn}] has an overall score below the Level 3 cutoff required for corporate placement drives.
                </p>
            </div>
            """, unsafe_allow_html=True)

    with col_metrics:
        st.markdown(f"""
        <div class="metric-tier-card">
            <div style="font-size:0.8rem; text-transform:uppercase; color:#c4b5fd; font-weight:600;">
                {svg_icon(ICO_LAYERS, '#c4b5fd', 15)} Assessment Tier Vector
            </div>
            <h1 style="color: #f5f3ff; margin: 4px 0 12px 0; font-size: 2.3rem;">LEVEL {overall}</h1>
            <div style="display:flex; flex-direction:column; gap:8px;">
                <div style="display:flex; justify-content:space-between;"><span style="color:#d8b4fe;">✦ Language:</span> <b>L{lx} / 4</b></div>
                <div style="display:flex; justify-content:space-between;"><span style="color:#a5b4fc;">✦ Aptitude:</span> <b>L{ax} / 4</b></div>
                <div style="display:flex; justify-content:space-between;"><span style="color:#e9d5ff;">✦ Core Test:</span> <b>L{cx} / 5</b></div>
                <div style="display:flex; justify-content:space-between;"><span style="color:#f0abfc;">✦ Programming:</span> <b>L{px} / 5</b></div>
                <div style="display:flex; justify-content:space-between;"><span style="color:#f472b6;">✦ Soft Skills:</span> <b>L{sx} / 4</b></div>
            </div>
            <div style="margin-top:14px; padding-top:10px; border-top:1px solid rgba(255,255,255,0.08); font-size:0.8rem; color:#94a3b8;">
                <div>10th Board: <b>{tenth_pct}%</b></div>
                <div>12th Board: <b>{twelfth_pct}%</b></div>
                <div>College CGPA: <b>{cgpa}</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)