import re
from datetime import datetime
from io import BytesIO

import pandas as pd
import plotly.express as px
import streamlit as st

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SkillGap AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM DESIGN
# ============================================================

st.markdown(
    """
    <style>

    /* MAIN PAGE */
    .stApp {
        background: linear-gradient(135deg, #101827, #172554, #312e81);
        color: #FFFFFF;
    }

    /* MAIN HEADINGS - HIGH VISIBILITY */
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4 {
    color: #FFFFFF !important;
    font-weight: 800 !important;
    text-shadow: 0 2px 8px rgba(0, 0, 0, 0.45) !important;
    letter-spacing: 0.4px !important;
}

/* MAIN PAGE TITLE */
.stApp h1 {
    font-size: 36px !important;
}

/* SECTION HEADINGS */
.stApp h2 {
    font-size: 28px !important;
}

/* SUBHEADINGS */
.stApp h3 {
    font-size: 22px !important;
}

/* SMALL HEADINGS */
.stApp h4 {
    font-size: 19px !important;
}

/* WHAT CAN YOU DO HEADING */
.stApp h3,
.stApp h2 {
    margin-bottom: 16px !important;
}

    /* NORMAL TEXT */
    p, label, li, span {
        color: #F1F5F9;
    }

    /* DASHBOARD HERO CARD */
    .hero-box {
        background: linear-gradient(120deg, #2563EB, #7C3AED);
        padding: 30px;
        border-radius: 22px;
        color: #FFFFFF !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.25);
        margin-bottom: 25px;
    }

    .hero-box h1,
    .hero-box h2,
    .hero-box h3,
    .hero-box p {
        color: #FFFFFF !important;
    }

    /* GENERAL CARDS */
    .card {
        background: #FFFFFF !important;
        padding: 22px;
        border-radius: 16px;
        color: #1E293B !important;
        box-shadow: 0 5px 18px rgba(0, 0, 0, 0.15);
        margin-bottom: 15px;
    }

    .card h3,
    .card p {
        color: #1E293B !important;
    }

    /* WHAT CAN YOU DO - DASHBOARD INFO BOXES */
    .info-box {
        background: #FFFFFF !important;
        color: #1E293B !important;
        padding: 24px !important;
        margin: 10px 0 !important;
        border: 2px solid #60A5FA !important;
        border-radius: 16px !important;
        min-height: 150px !important;
        box-shadow: 0 5px 18px rgba(37, 99, 235, 0.15) !important;
        transition: all 0.3s ease !important;
    }

    .info-box h3 {
        color: #1D4ED8 !important;
        font-size: 20px !important;
        font-weight: 800 !important;
        margin-top: 0 !important;
        margin-bottom: 12px !important;
    }

    .info-box p {
        color: #334155 !important;
        font-size: 15px !important;
        line-height: 1.7 !important;
    }

    .info-box:hover {
        border-color: #7C3AED !important;
        box-shadow: 0 8px 22px rgba(124, 58, 237, 0.2) !important;
        transform: translateY(-3px);
    }

    /* BUTTON DESIGN */
    .stButton > button,
    .stDownloadButton > button {
        background: linear-gradient(135deg, #2563EB, #7C3AED) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 18px !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.20) !important;
    }

    /* BUTTON TEXT */
    .stButton > button p,
    .stDownloadButton > button p,
    .stButton > button span,
    .stDownloadButton > button span {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* BUTTON HOVER */
    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: #1D4ED8 !important;
        color: #FFFFFF !important;
        border: 2px solid #93C5FD !important;
    }

    /* DOWNLOAD BUTTON ICON */
    .stDownloadButton > button svg {
        fill: #FFFFFF !important;
    }

    /* INPUT BOXES */
    .stTextInput input,
    .stTextArea textarea,
    .stNumberInput input {
        background-color: #FFFFFF !important;
        color: #111827 !important;
        border: 2px solid #93C5FD !important;
        border-radius: 8px !important;
    }

    /* DROPDOWNS */
    .stSelectbox div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        color: #111827 !important;
        border-radius: 8px !important;
    }

    /* RESUME UPLOAD AREA */
    [data-testid="stFileUploader"] {
        background: #DBEAFE !important;
        padding: 20px !important;
        border: 2px dashed #2563EB !important;
        border-radius: 14px !important;
    }

    /* UPLOAD AREA TEXT */
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] small {
        color: #1E3A8A !important;
        font-weight: 600 !important;
    }

    /* BROWSE FILES BUTTON */
    [data-testid="stFileUploader"] button {
        background: #2563EB !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: bold !important;
    }

    [data-testid="stFileUploader"] button:hover {
        background: #1D4ED8 !important;
        color: #FFFFFF !important;
    }

    /* METRIC CARDS */
    [data-testid="stMetric"] {
        background: #FFFFFF !important;
        padding: 18px !important;
        border-radius: 14px !important;
        border: 2px solid #60A5FA !important;
    }

    /* METRIC LABELS */
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricLabel"] div {
        color: #334155 !important;
        font-size: 16px !important;
        font-weight: 700 !important;
    }

    /* METRIC VALUES */
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div,
    [data-testid="stMetricValue"] p {
        color: #1D4ED8 !important;
        font-size: 30px !important;
        font-weight: 800 !important;
    }

    /* METRIC DELTA */
    [data-testid="stMetricDelta"],
    [data-testid="stMetricDelta"] div {
        color: #15803D !important;
    }

    /* TABLES */
    [data-testid="stDataFrame"] {
        background: #FFFFFF !important;
        border-radius: 12px !important;
    }

    /* PROGRESS BAR */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #2563EB, #A855F7);
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827, #1E3A8A) !important;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div {
        color: #FFFFFF !important;
    }

    /* ALERTS */
    [data-testid="stAlert"] {
        border-radius: 12px !important;
    }

    /* EXPANDERS */
    [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 12px !important;
        border: 1px solid #64748B !important;
    }

    /* PAGE LAYOUT */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)
# ============================================================
# CAREER SKILL DATABASE
# ============================================================

career_skills = {
    "Python Developer": [
        "Python",
        "SQL",
        "Git",
        "Data Structures",
        "Problem Solving",
        "OOP",
        "APIs",
        "Testing",
    ],

    "Machine Learning Engineer": [
        "Python",
        "Machine Learning",
        "Statistics",
        "NumPy",
        "Pandas",
        "Scikit-learn",
        "SQL",
        "Data Visualization",
    ],

    "Data Scientist": [
        "Python",
        "Statistics",
        "Machine Learning",
        "SQL",
        "Pandas",
        "NumPy",
        "Data Visualization",
        "Data Cleaning",
    ],

    "Data Analyst": [
        "SQL",
        "Excel",
        "Python",
        "Power BI",
        "Statistics",
        "Data Visualization",
        "Data Cleaning",
    ],

    "AI Engineer": [
        "Python",
        "Machine Learning",
        "Deep Learning",
        "Statistics",
        "NLP",
        "TensorFlow",
        "PyTorch",
        "Data Processing",
    ],

    "Web Developer": [
        "HTML",
        "CSS",
        "JavaScript",
        "Python",
        "SQL",
        "APIs",
        "Git",
        "Responsive Design",
    ],

    "Full Stack Developer": [
        "HTML",
        "CSS",
        "JavaScript",
        "Python",
        "SQL",
        "APIs",
        "Git",
        "Backend Development",
        "Frontend Development",
    ],

    "Cybersecurity Analyst": [
        "Networking",
        "Linux",
        "Python",
        "Cybersecurity",
        "Security Analysis",
        "Operating Systems",
        "Problem Solving",
    ],

    "Cloud Engineer": [
        "Linux",
        "Networking",
        "Python",
        "Cloud Computing",
        "Docker",
        "Git",
        "Security",
        "Automation",
    ],
}


# ============================================================
# SKILL KEYWORD ALIASES
# ============================================================

skill_aliases = {
    "Python": [
        "python",
    ],

    "SQL": [
        "sql",
        "structured query language",
    ],

    "Git": [
        "git",
        "github",
        "version control",
    ],

    "Data Structures": [
        "data structures",
        "data structure",
    ],

    "Problem Solving": [
        "problem solving",
        "problem-solving",
        "logical thinking",
    ],

    "OOP": [
        "oop",
        "object oriented programming",
        "object-oriented programming",
    ],

    "APIs": [
        "api",
        "apis",
        "rest api",
        "restful api",
    ],

    "Testing": [
        "software testing",
        "unit testing",
        "testing",
    ],

    "Machine Learning": [
        "machine learning",
        "machine-learning",
        "ml",
    ],

    "Statistics": [
        "statistics",
        "statistical analysis",
    ],

    "NumPy": [
        "numpy",
    ],

    "Pandas": [
        "pandas",
    ],

    "Scikit-learn": [
        "scikit-learn",
        "sklearn",
    ],

    "Data Visualization": [
        "data visualization",
        "data visualisation",
        "visualization",
        "visualisation",
    ],

    "Data Cleaning": [
        "data cleaning",
        "data cleansing",
        "data preprocessing",
        "data cleaning and preprocessing",
    ],

    "Excel": [
        "excel",
        "microsoft excel",
    ],

    "Power BI": [
        "power bi",
    ],

    "Deep Learning": [
        "deep learning",
    ],

    "NLP": [
        "natural language processing",
        "nlp",
    ],

    "TensorFlow": [
        "tensorflow",
    ],

    "PyTorch": [
        "pytorch",
    ],

    "Data Processing": [
        "data processing",
        "data preprocessing",
    ],

    "HTML": [
        "html",
        "html5",
    ],

    "CSS": [
        "css",
        "css3",
    ],

    "JavaScript": [
        "javascript",
        "java script",
    ],

    "Responsive Design": [
        "responsive design",
        "mobile responsive",
    ],

    "Networking": [
        "networking",
        "computer networks",
        "network security",
    ],

    "Linux": [
        "linux",
    ],

    "Cybersecurity": [
        "cybersecurity",
        "cyber security",
        "information security",
    ],

    "Security Analysis": [
        "security analysis",
        "vulnerability assessment",
    ],

    "Operating Systems": [
        "operating systems",
        "operating system",
        "os concepts",
    ],

    "Cloud Computing": [
        "cloud computing",
        "aws",
        "azure",
        "google cloud",
    ],

    "Docker": [
        "docker",
        "containerization",
    ],

    "Security": [
        "security",
        "secure coding",
    ],

    "Automation": [
        "automation",
        "scripting",
    ],

    "Backend Development": [
        "backend development",
        "back-end development",
        "server side development",
    ],

    "Frontend Development": [
        "frontend development",
        "front-end development",
        "client side development",
    ],
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_skills(text):
    """
    Detect known skill keywords in text.

    This is keyword-based extraction, not a trained AI model.
    """

    if not text:
        return []

    text = text.lower()
    found_skills = []

    for skill, aliases in skill_aliases.items():

        for alias in aliases:

            pattern = (
                r"(?<![a-zA-Z0-9+#.])"
                + re.escape(alias.lower())
                + r"(?![a-zA-Z0-9+#.])"
            )

            if re.search(pattern, text):
                found_skills.append(skill)
                break

    return sorted(set(found_skills))


def extract_pdf_text(uploaded_file):
    """
    Extract selectable text from an uploaded PDF.

    Scanned PDFs may require OCR and might return no text.
    """

    if uploaded_file is None:
        return ""

    try:
        uploaded_file.seek(0)

        reader = PdfReader(uploaded_file)

        pages = []

        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages.append(page_text)

        return "\n".join(pages)

    except Exception as error:
        st.error(f"Unable to read this PDF: {error}")
        return ""


def compare_skills(required_skills, user_skills):
    """
    Compare required skills with skills detected in user text.
    """

    required = set(required_skills)
    available = set(user_skills)

    matched = sorted(required.intersection(available))
    missing = sorted(required.difference(available))

    if not required:
        match_score = 0
    else:
        match_score = round(
            (len(matched) / len(required)) * 100,
            2,
        )

    return {
        "matched": matched,
        "missing": missing,
        "score": match_score,
    }


def calculate_text_similarity(resume_text, job_text):
    """
    Calculate text similarity using TF-IDF and cosine similarity.

    This measures textual similarity. It does not guarantee
    job suitability or predict hiring decisions.
    """

    if not resume_text.strip() or not job_text.strip():
        return 0.0

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )

        matrix = vectorizer.fit_transform(
            [resume_text, job_text]
        )

        similarity = cosine_similarity(
            matrix[0:1],
            matrix[1:2],
        )[0][0]

        return round(float(similarity) * 100, 2)

    except ValueError:
        return 0.0

def recommend_careers(user_skills, user_interests, preferred_area):

    results = []

    interest_mapping = {
        "Programming": [
            "Python Developer",
            "Web Developer",
            "Full Stack Developer",
        ],
        "Artificial Intelligence": [
            "AI Engineer",
            "Machine Learning Engineer",
            "Data Scientist",
        ],
        "Data Analysis": [
            "Data Analyst",
            "Data Scientist",
            "Machine Learning Engineer",
        ],
        "Web Development": [
            "Web Developer",
            "Full Stack Developer",
        ],
        "Cybersecurity": [
            "Cybersecurity Analyst",
        ],
        "Cloud Computing": [
            "Cloud Engineer",
        ],
    }

    for career, required_skills in career_skills.items():

        required = set(required_skills)
        available = set(user_skills)

        matched = sorted(required.intersection(available))
        missing = sorted(required.difference(available))

        skill_score = (
            len(matched) / len(required) * 100
            if required else 0
        )

        interest_matches = sum(
            career in interest_mapping.get(interest, [])
            for interest in user_interests
        )

        interest_score = (
            min(100, interest_matches / len(user_interests) * 100)
            if user_interests else 0
        )

        area_careers = {
            "Software Development": [
                "Python Developer",
                "Web Developer",
                "Full Stack Developer",
            ],
            "AI and Data": [
                "AI Engineer",
                "Machine Learning Engineer",
                "Data Scientist",
                "Data Analyst",
            ],
            "Security": [
                "Cybersecurity Analyst",
            ],
            "Cloud": [
                "Cloud Engineer",
            ],
        }

        if preferred_area == "Any Area":
            area_score = 100
        else:
            area_score = (
                100
                if career in area_careers.get(preferred_area, [])
                else 0
            )

        final_score = (
            skill_score * 0.60
            + interest_score * 0.25
            + area_score * 0.15
        )

        results.append({
            "Career": career,
            "Recommendation Score": round(final_score, 2),
            "Skill Match": round(skill_score, 2),
            "Matched Skills": matched,
            "Missing Skills": missing,
        })

    results.sort(
        key=lambda item: item["Recommendation Score"],
        reverse=True,
    )

    return results
def create_roadmap(missing_skills):
    """
    Create a simple learning roadmap from missing skills.
    """

    roadmap = []

    for index, skill in enumerate(missing_skills, start=1):

        roadmap.append({
            "Week": f"Week {index}",
            "Skill to Learn": skill,
            "Suggested Activity": (
                f"Learn the fundamentals of {skill}, "
                f"complete a small practical exercise, "
                f"and build a mini-project using {skill}."
            ),
        })

    return roadmap


def generate_career_report(
    career,
    matched_skills,
    missing_skills,
    similarity_score=None,
    skill_match_score=None,
):
    """
    Generate a downloadable PDF report.
    """

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=23,
        leading=28,
        textColor=colors.HexColor("#1D4ED8"),
        spaceAfter=12,
    )

    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#1D4ED8"),
        spaceBefore=14,
        spaceAfter=8,
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontSize=10,
        leading=15,
    )

    story = []

    story.append(
        Paragraph("SkillGap AI", title_style)
    )

    story.append(
        Paragraph(
            "Career and Skill Gap Analysis Report",
            styles["Heading2"],
        )
    )

    story.append(Spacer(1, 10))

    story.append(
        Paragraph(
            f"<b>Analysis:</b> {career}",
            body_style,
        )
    )

    story.append(
        Paragraph(
            f"<b>Generated on:</b> "
            f"{datetime.now().strftime('%d %B %Y, %I:%M %p')}",
            body_style,
        )
    )

    story.append(Spacer(1, 12))

    if similarity_score is not None:
        story.append(
            Paragraph(
                f"<b>Resume–job text similarity:</b> "
                f"{similarity_score}%",
                body_style,
            )
        )

    if skill_match_score is not None:
        story.append(
            Paragraph(
                f"<b>Required skill match:</b> "
                f"{skill_match_score}%",
                body_style,
            )
        )

    story.append(
        Paragraph("Matched Skills", heading_style)
    )

    if matched_skills:

        matched_data = [
            ["No.", "Skill"],
        ]

        for index, skill in enumerate(matched_skills, start=1):
            matched_data.append([str(index), skill])

        matched_table = Table(
            matched_data,
            colWidths=[0.65 * inch, 5.8 * inch],
            repeatRows=1,
        )

        matched_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#2563EB"),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor("#D1D5DB"),
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
            ])
        )

        story.append(matched_table)

    else:
        story.append(
            Paragraph(
                "No matching skills were detected.",
                body_style,
            )
        )

    story.append(
        Paragraph("Skills to Improve", heading_style)
    )

    if missing_skills:

        missing_data = [
            ["No.", "Skill to Learn"],
        ]

        for index, skill in enumerate(missing_skills, start=1):
            missing_data.append([str(index), skill])

        missing_table = Table(
            missing_data,
            colWidths=[0.65 * inch, 5.8 * inch],
            repeatRows=1,
        )

        missing_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#7C3AED"),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor("#D1D5DB"),
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
            ])
        )

        story.append(missing_table)

    else:
        story.append(
            Paragraph(
                "No missing skills were identified "
                "among the skills checked.",
                body_style,
            )
        )

    story.append(
        Paragraph("Recommended Next Steps", heading_style)
    )

    recommendations = [
        "Study the missing skills in a structured order.",
        "Complete practical exercises for each skill.",
        "Build projects that demonstrate your abilities.",
        "Update your resume with skills and projects you "
        "can genuinely demonstrate.",
    ]

    for recommendation in recommendations:
        story.append(
            Paragraph(
                f"&bull; {recommendation}",
                body_style,
            )
        )
        story.append(Spacer(1, 5))

    story.append(Spacer(1, 15))

    story.append(
        Paragraph(
            "<b>Important note:</b> This report is based on "
            "the text and known skill keywords provided. "
            "Text similarity and skill overlap are indicators, "
            "not guarantees of job suitability, interview "
            "selection, or employment.",
            body_style,
        )
    )

    document.build(story)

    buffer.seek(0)

    return buffer.getvalue()


def show_skill_lists(matched, missing):
    """
    Display matched and missing skills.
    """

    left, right = st.columns(2)

    with left:
        st.subheader("✅ Matched Skills")

        if matched:
            for skill in matched:
                st.success(skill)
        else:
            st.info("No matched skills detected.")

    with right:
        st.subheader("📚 Skills to Improve")

        if missing:
            for skill in missing:
                st.warning(skill)
        else:
            st.success("No missing skills detected in this comparison.")


def display_skill_comparison(required_skills, detected_skills):
    """
    Show skill comparison and a progress chart.
    """

    comparison = compare_skills(
        required_skills,
        detected_skills,
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Skill Match",
        f"{comparison['score']}%",
    )

    col2.metric(
        "Matched Skills",
        len(comparison["matched"]),
    )

    col3.metric(
        "Missing Skills",
        len(comparison["missing"]),
    )

    st.progress(
        int(comparison["score"])
    )

    show_skill_lists(
        comparison["matched"],
        comparison["missing"],
    )

    return comparison


# ============================================================
# SESSION STATE
# ============================================================

if "matcher_result" not in st.session_state:
    st.session_state.matcher_result = None

if "last_roadmap" not in st.session_state:
    st.session_state.last_roadmap = []


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🎯 SkillGap AI")

st.sidebar.caption(
    "Intelligent Career & Skill Gap Analyzer"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Skill Gap Analyzer",
        "Job Description Analyzer",
        "Resume Analyzer",
        "Resume–Job Matcher",
        "Learning Roadmap",
        "Career Comparison",
        "Career Recommendations",
        "About Project",
        
    ],
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Explore career requirements, compare skills, "
    "and plan your learning journey."
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown(
        """
        <div class="hero-box">
            <h1>🎯 SkillGap AI</h1>
            <p>
                Understand your skills, discover career gaps,
                and build a structured learning roadmap.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Your Career Development Workspace")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Career Profiles",
        len(career_skills),
    )

    col2.metric(
        "Analysis Tools",
        4,
    )

    col3.metric(
        "Core Technologies",
        6,
    )

    st.markdown("### 🚀 What can you do?")

    a, b = st.columns(2)

    with a:
        st.markdown(
            """
            <div class="info-box">
                <h3>🧭 Career Skill Analysis</h3>
                Compare your current skills against
                the requirements of a target career.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="info-box">
                <h3>📄 Resume Analysis</h3>
                Upload a text-based PDF resume to
                identify recognized technical skills.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with b:
        st.markdown(
            """
            <div class="info-box">
                <h3>🔍 Resume–Job Matching</h3>
                Compare resume content with a job
                description using TF-IDF text similarity.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="info-box">
                <h3>🗺️ Learning Roadmap</h3>
                Organize missing skills into a
                simple week-by-week learning plan.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.caption(
        "Note: Skill detection uses known keywords. "
        "Resume–job similarity uses TF-IDF and cosine "
        "similarity. These tools do not guarantee hiring outcomes."
    )


# ============================================================
# SKILL GAP ANALYZER
# ============================================================

elif page == "Skill Gap Analyzer":

    st.title("🧭 Skill Gap Analyzer")

    st.write(
        "Choose a target career and select the skills "
        "you already know."
    )

    career = st.selectbox(
        "Select Target Career",
        list(career_skills.keys()),
    )

    required_skills = career_skills[career]

    st.subheader(
        f"Skills commonly associated with {career}"
    )

    st.write(", ".join(required_skills))

    selected_skills = st.multiselect(
        "Select your existing skills",
        options=sorted(skill_aliases.keys()),
    )

    if st.button(
        "Analyze My Skill Gap",
        type="primary",
    ):

        comparison = display_skill_comparison(
            required_skills,
            selected_skills,
        )

        st.session_state.last_roadmap = create_roadmap(
            comparison["missing"]
        )

        st.success(
            "Analysis completed. Open Learning Roadmap "
            "to review your next steps."
        )


# ============================================================
# JOB DESCRIPTION ANALYZER
# ============================================================

elif page == "Job Description Analyzer":

    st.title("📋 Job Description Analyzer")

    st.write(
        "Paste a job description to identify recognized "
        "technical skills and compare them with a career profile."
    )

    job_text = st.text_area(
        "Paste Job Description",
        height=250,
        placeholder=(
            "Example: We are looking for a Python developer "
            "with SQL, Git, data structures, and API experience."
        ),
    )

    selected_career = st.selectbox(
        "Compare against career profile",
        list(career_skills.keys()),
        key="job_career",
    )

    if st.button(
        "Analyze Job Description",
        type="primary",
    ):

        if not job_text.strip():
            st.warning(
                "Please paste a job description first."
            )

        else:

            detected_skills = extract_skills(job_text)

            st.subheader("Recognized Skills")

            if detected_skills:
                st.write(", ".join(detected_skills))
            else:
                st.info(
                    "No recognized skills were found. "
                    "Try including more specific skill names."
                )

            st.subheader("Comparison with Career Profile")

            display_skill_comparison(
                career_skills[selected_career],
                detected_skills,
            )


# ============================================================
# RESUME ANALYZER
# ============================================================

elif page == "Resume Analyzer":

    st.title("📄 Resume Analyzer")

    st.write(
        "Upload a PDF resume to identify known skills "
        "and compare them with a target career."
    )

    uploaded_resume = st.file_uploader(
        "Upload Resume PDF",
        type=["pdf"],
        key="resume_analyzer_upload",
    )

    selected_career = st.selectbox(
        "Choose Target Career",
        list(career_skills.keys()),
        key="resume_career",
    )

    if st.button(
        "Analyze Resume",
        type="primary",
    ):

        if uploaded_resume is None:

            st.warning(
                "Please upload a PDF resume first."
            )

        else:

            resume_text = extract_pdf_text(
                uploaded_resume
            )

            if not resume_text.strip():

                st.error(
                    "No readable text was found in this PDF. "
                    "It may be scanned or image-based. "
                    "Try a text-based PDF."
                )

            else:

                detected_skills = extract_skills(
                    resume_text
                )

                st.subheader("Resume Analysis Results")

                st.metric(
                    "Recognized Skills",
                    len(detected_skills),
                )

                if detected_skills:
                    st.write(
                        ", ".join(detected_skills)
                    )
                else:
                    st.info(
                        "No known skills were detected "
                        "using the current keyword list."
                    )

                st.subheader("Target Career Comparison")

                comparison = display_skill_comparison(
                    career_skills[selected_career],
                    detected_skills,
                )

                report_pdf = generate_career_report(
                    career=selected_career,
                    matched_skills=comparison["matched"],
                    missing_skills=comparison["missing"],
                    skill_match_score=comparison["score"],
                )

                st.download_button(
                    label="📥 Download Career Analysis Report",
                    data=report_pdf,
                    file_name="SkillGapAI_Resume_Report.pdf",
                    mime="application/pdf",
                )

                st.session_state.last_roadmap = create_roadmap(
                    comparison["missing"]
                )


# ============================================================
# RESUME–JOB MATCHER
# ============================================================

elif page == "Resume–Job Matcher":

    st.title("🔍 Resume–Job Matcher")

    st.write(
        "Compare your resume with a job description. "
        "The application calculates text similarity and "
        "compares recognized skills."
    )

    uploaded_resume = st.file_uploader(
        "Upload Your Resume (PDF)",
        type=["pdf"],
        key="matcher_resume_upload",
    )

    job_description = st.text_area(
        "Paste Job Description",
        height=230,
        placeholder=(
            "Paste the complete job description here..."
        ),
        key="matcher_job_description",
    )

    if st.button(
        "Match Resume with Job",
        type="primary",
    ):

        if uploaded_resume is None:

            st.warning(
                "Please upload your resume."
            )

            st.session_state.matcher_result = None

        elif not job_description.strip():

            st.warning(
                "Please paste the job description."
            )

            st.session_state.matcher_result = None

        else:

            resume_text = extract_pdf_text(
                uploaded_resume
            )

            if not resume_text.strip():

                st.error(
                    "No readable text was found in the PDF. "
                    "Please try a text-based PDF."
                )

                st.session_state.matcher_result = None

            else:

                resume_skills = extract_skills(
                    resume_text
                )

                job_skills = extract_skills(
                    job_description
                )

                skill_comparison = compare_skills(
                    job_skills,
                    resume_skills,
                )

                similarity = calculate_text_similarity(
                    resume_text,
                    job_description,
                )

                st.session_state.matcher_result = {
                    "similarity": similarity,
                    "skill_score": skill_comparison["score"],
                    "matched": skill_comparison["matched"],
                    "missing": skill_comparison["missing"],
                    "resume_skills": resume_skills,
                    "job_skills": job_skills,
                }

                st.session_state.last_roadmap = create_roadmap(
                    skill_comparison["missing"]
                )

    result = st.session_state.matcher_result

    if result:

        st.markdown("---")

        st.subheader("📊 Matching Results")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Text Similarity",
            f"{result['similarity']}%",
        )

        col2.metric(
            "Skill Match",
            f"{result['skill_score']}%",
        )

        col3.metric(
            "Matched Skills",
            len(result["matched"]),
        )

        st.progress(
            int(result["skill_score"])
        )

        st.caption(
            "Text similarity compares the wording of the two texts. "
            "Skill match measures the overlap between recognized "
            "skills in the job description and resume. "
            "Neither score represents a hiring probability."
        )

        show_skill_lists(
            result["matched"],
            result["missing"],
        )

        st.markdown("---")

        st.subheader("📄 Download Your Analysis")

        report_pdf = generate_career_report(
            career="Resume–Job Match Analysis",
            matched_skills=result["matched"],
            missing_skills=result["missing"],
            similarity_score=result["similarity"],
            skill_match_score=result["skill_score"],
        )

        st.download_button(
            label="📥 Download Career Analysis Report",
            data=report_pdf,
            file_name="SkillGapAI_Career_Report.pdf",
            mime="application/pdf",
        )

        st.subheader("🗺️ Suggested Learning Roadmap")

        roadmap = create_roadmap(
            result["missing"]
        )

        if roadmap:

            roadmap_df = pd.DataFrame(roadmap)

            st.dataframe(
                roadmap_df,
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.success(
                "No missing recognized skills were identified."
            )


# ============================================================
# LEARNING ROADMAP
# ============================================================

elif page == "Learning Roadmap":

    st.title("🗺️ Personalized Learning Roadmap")

    st.write(
        "Create a learning plan based on the skills "
        "you want to develop."
    )

    roadmap_source = st.radio(
        "Choose roadmap source",
        [
            "Use my previous analysis",
            "Select skills manually",
        ],
    )

    if roadmap_source == "Use my previous analysis":

        if st.session_state.last_roadmap:

            roadmap = st.session_state.last_roadmap

            st.success(
                "Your roadmap is based on your latest analysis."
            )

        else:

            st.info(
                "No previous analysis is available in this session. "
                "Run a Skill Gap Analysis or Resume Analysis first, "
                "or choose manual selection."
            )

            roadmap = []

    else:

        selected_missing = st.multiselect(
            "Select the skills you want to learn",
            options=sorted(skill_aliases.keys()),
        )

        roadmap = create_roadmap(
            selected_missing
        )

    if roadmap:

        st.subheader("Your Learning Plan")

        roadmap_df = pd.DataFrame(roadmap)

        st.dataframe(
            roadmap_df,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Recommended Learning Process")

        st.markdown(
            """
            1. **Learn:** Understand the basic concepts.
            2. **Practice:** Solve small exercises.
            3. **Build:** Create a small project.
            4. **Review:** Test your understanding.
            5. **Document:** Add genuine work to your portfolio.
            """
        )

        csv_data = roadmap_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="📥 Download Learning Roadmap (CSV)",
            data=csv_data,
            file_name="SkillGapAI_Learning_Roadmap.csv",
            mime="text/csv",
        )


# ============================================================
# CAREER COMPARISON
# ============================================================

elif page == "Career Comparison":

    st.title("⚖️ Career Comparison")

    st.write(
        "Compare the skills commonly associated with two careers."
    )

    col1, col2 = st.columns(2)

    with col1:

        career_a = st.selectbox(
            "First Career",
            list(career_skills.keys()),
            index=0,
        )

    with col2:

        career_b = st.selectbox(
            "Second Career",
            list(career_skills.keys()),
            index=1,
        )

    if career_a == career_b:

        st.info(
            "Choose two different careers for a comparison."
        )

    else:

        skills_a = set(
            career_skills[career_a]
        )

        skills_b = set(
            career_skills[career_b]
        )

        common_skills = sorted(
            skills_a.intersection(skills_b)
        )

        only_a = sorted(
            skills_a.difference(skills_b)
        )

        only_b = sorted(
            skills_b.difference(skills_a)
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Common Skills",
            len(common_skills),
        )

        col2.metric(
            f"{career_a} Skills",
            len(skills_a),
        )

        col3.metric(
            f"{career_b} Skills",
            len(skills_b),
        )

        if common_skills:

            st.subheader("🤝 Skills Common to Both Careers")

            st.write(
                ", ".join(common_skills)
            )

        left, right = st.columns(2)

        with left:

            st.subheader(
                f"Skills for {career_a}"
            )

            for skill in only_a:
                st.write(f"• {skill}")

        with right:

            st.subheader(
                f"Skills for {career_b}"
            )

            for skill in only_b:
                st.write(f"• {skill}")

        chart_df = pd.DataFrame({
            "Career": [
                career_a,
                career_b,
            ],
            "Number of Skills": [
                len(skills_a),
                len(skills_b),
            ],
        })

        fig = px.bar(
            chart_df,
            x="Career",
            y="Number of Skills",
            title="Career Skill Profile Comparison",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        # ============================================================
# ABOUT PROJECT
# ============================================================
elif page == "Career Recommendations":

    st.title("🎯 Career Recommendation System")

    st.write(
        "Discover careers that may suit your current skills, "
        "interests, and preferred work area."
    )

    st.info(
        "Choose your skills and interests to explore "
        "career options and identify skills you can develop."
    )

    user_skills = st.multiselect(
        "Select the skills you currently know",
        options=sorted(skill_aliases.keys()),
        key="career_recommendation_skills",
    )

    user_interests = st.multiselect(
        "Select your interests",
        [
            "Programming",
            "Artificial Intelligence",
            "Data Analysis",
            "Web Development",
            "Cybersecurity",
            "Cloud Computing",
        ],
        key="career_recommendation_interests",
    )

    preferred_area = st.selectbox(
        "Choose your preferred career area",
        [
            "Any Area",
            "Software Development",
            "AI and Data",
            "Security",
            "Cloud",
        ],
        key="career_recommendation_area",
    )

    if st.button("🚀 Find My Best Careers", type="primary"):

        if not user_skills and not user_interests:
            st.warning(
                "Please select at least one skill or interest."
            )

        else:
            recommendations = recommend_careers(
                user_skills,
                user_interests,
                preferred_area,
            )

            st.subheader("Your Career Recommendations")

            st.caption(
                "Scores are calculated from your selected skills, "
                "interests, and career preferences. They are guidance "
                "scores, not predictions of employment success."
            )

            for index, item in enumerate(
                recommendations[:5],
                start=1,
            ):

                with st.container(border=True):

                    st.subheader(
                        f"{index}. {item['Career']}"
                    )

                    col1, col2 = st.columns(2)

                    col1.metric(
                        "Recommendation Score",
                        f"{item['Recommendation Score']}%",
                    )

                    col2.metric(
                        "Skill Match",
                        f"{item['Skill Match']}%",
                    )

                    st.progress(
                        int(item["Recommendation Score"])
                    )

                    st.write("**Matched Skills**")

                    if item["Matched Skills"]:
                        st.write(
                            ", ".join(item["Matched Skills"])
                        )
                    else:
                        st.write(
                            "No matching skills found yet."
                        )

                    st.write("**Skills to Develop**")

                    if item["Missing Skills"]:
                        st.write(
                            ", ".join(item["Missing Skills"])
                        )
                    else:
                        st.write(
                            "You match all configured skills "
                            "for this career."
                        )

                    st.write(
                        f"**Your current match:** "
                        f"{len(item['Matched Skills'])} out of "
                        f"{len(career_skills[item['Career']])} "
                        f"configured skills."
                    )
elif page == "About Project":

    st.title("ℹ️ About SkillGap AI")

    st.markdown(
        """
        ### Project Overview

        **SkillGap AI — Intelligent Career & Skill Gap Analyzer**
        is a career-planning application designed to help users
        understand recognized skills, compare career requirements,
        and organize learning goals.

        ### Technologies Used

        - **Python:** Application logic and processing.
        - **Streamlit:** Interactive web interface.
        - **Pandas:** Tabular data and roadmap handling.
        - **Scikit-learn:** TF-IDF feature extraction and
          cosine similarity.
        - **Plotly:** Interactive charts.
        - **pypdf:** Text extraction from PDF resumes.
        - **ReportLab:** Downloadable PDF analysis reports.

        ### Main Features

        1. Career skill gap analysis.
        2. Job description keyword analysis.
        3. PDF resume skill analysis.
        4. Resume–job text similarity.
        5. Missing-skill identification.
        6. Learning roadmap generation.
        7. Career profile comparison.
        8. Downloadable career analysis report.

        ### How the Analysis Works

        The application identifies skills using a predefined
        vocabulary of skill names and aliases.

        For resume–job matching, TF-IDF converts text into
        numerical features. Cosine similarity then measures
        the similarity between the resume and job-description
        text.

        ### Current Limitations

        - Skill detection depends on the predefined keyword list.
        - Scanned or image-only PDFs may not provide readable text.
        - Text similarity is not a hiring probability.
        - Career profiles are illustrative, not universal job standards.
        - Recommendations should be reviewed and personalized
          by the user.

        ### Project Goal

        Help students make more informed learning plans and
        understand the skills they may need to develop for
        different career paths.
        """
    )

    st.markdown("---")

    st.caption(
        "SkillGap AI is a career-planning aid. "
        "It does not guarantee employment or selection."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#9ca3af;">
        SkillGap AI | Intelligent Career & Skill Gap Analyzer
        <br>
        Learn • Analyze • Improve
    </div>
    """,
    unsafe_allow_html=True,
)