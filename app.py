import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os

# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load("Stacking_Model.joblib")


# ==========================================
# PAGE SETTINGS
# ==========================================

TITLE = "FUOYE MOOC Engagement & Completion Prediction System"

DESCRIPTION = """
Predict students at risk of dropping out using Machine Learning
and Explainable Artificial Intelligence (XAI).
"""


# ==========================================
# LOGOS
# ==========================================

FUOYE_LOGO = "assets/fuoye_logo.png"
EA_LOGO = "assets/ea_logo.png"
# ==========================================
# ENCODING TABLES
# ==========================================

STUDY_MAP = {
    "0-2 hours": 0,
    "3-5 hours": 1,
    "6-8 hours": 2,
    "9+ hours": 3
}

INTERNET_MAP = {
    1: 0.00,
    2: 0.25,
    3: 0.50,
    4: 0.75,
    5: 1.00,
}
ELECTRICITY_MAP = {
    "More than 12 hours": 0.00,
    "9-12 hours": 0.25,
    "5-8 hours": 0.50,
    "2-4 hours": 0.75,
    "Less than 2 hours": 1.00,
}
DATA_MAP = {
    1: 0.00,
    2: 0.25,
    3: 0.50,
    4: 0.75,
    5: 1.00,
}

FORUM_MAP = {
    "Never": 0,
    "Rarely": 1,
    "Sometimes": 2,
    "Often": 3,
    "Very Often": 4
}

ASSIGNMENT_MAP = {
    "0-25%": 0,
    "26-50%": 1,
    "51-75%": 2,
    "76-100%": 3
}

VIDEO_MAP = {
    "0-25%": 0,
    "26-50%": 1,
    "51-75%": 2,
    "76-100%": 3
}

DECLARED_MAP = {
    "Unsure": 0,
    "Intended to audit specific modules without completion": 0,
    "Intended to complete and earn certificate": 1
}

INCOME_MAP = {
    "Under ₦20,000": 0,
    "₦20,000 - ₦50,000": 1,
    "₦51,000 - ₦100,000": 2,
    "Over ₦100,000": 3
}

NIGHT_MAP = {
    "No, never": 0,
    "No, rarely": 1,
    "Yes, frequently": 2,
    "Yes, exclusively": 3
}
# ==========================================
# FEATURE ENGINEERING
# ==========================================

def calculate_mai(motivation_score, barrier_score):
    """
    Motivation–Anxiety Index
    Higher motivation and lower barriers produce a better score.
    """
    return motivation_score - barrier_score


def calculate_ics(internet,
                  electricity,
                  data_affordability):

    internet = INTERNET_MAP[internet]

    electricity = ELECTRICITY_MAP[electricity]

    data = DATA_MAP[data_affordability]

    return (
        internet +
        electricity +
        data
    ) / 3


def calculate_income_nightdata(income, night_data):
    return income * night_data


def calculate_mai_ics(mai, ics):
    return mai * ics


def calculate_nightdata_motivation(night_data, motivation):
    return night_data * motivation
# ==========================================
# RECOMMENDATION ENGINE
# ==========================================

def generate_recommendations(
        study,
        forum,
        assignments,
        motivation,
        barrier,
        ics):

    recommendations = []

    if study <= 1:
        recommendations.append(
            "📚 Increase weekly study hours.")

    if forum <= 1:
        recommendations.append(
            "💬 Participate more actively in discussion forums.")

    if assignments <= 1:
        recommendations.append(
            "📝 Complete assignments consistently.")

    if motivation < 3:
        recommendations.append(
            "🎯 Improve learning motivation through academic mentoring.")

    if barrier > 3:
        recommendations.append(
            "⚠ Address learning barriers affecting course engagement.")

    if ics >= 0.70:
        recommendations.append(
            "🌐 Improve internet and electricity access where possible.")

    if len(recommendations) == 0:
        recommendations.append(
            "✅ Student demonstrates healthy engagement. Continue current learning behaviour.")

    return "\n".join(recommendations)
# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_dropout(
        study,
        forum,
        assignment,
        video,
        intention,
        digital,
        motivation,
        barrier,
        internet,
        electricity,
        data_affordability,
        income,
        night):

    # --------------------------
    # Encode categorical inputs
    # --------------------------

    study = STUDY_MAP[study]
    forum = FORUM_MAP[forum]
    assignment = ASSIGNMENT_MAP[assignment]
    video = VIDEO_MAP[video]
    intention = DECLARED_MAP[intention]
    income = INCOME_MAP[income]
    night = NIGHT_MAP[night]

    # --------------------------
    # Feature Engineering
    # --------------------------

    mai = calculate_mai(
        motivation,
        barrier
    )

    ics = calculate_ics(
        internet,
        electricity,
        data_affordability
    )

    income_night = calculate_income_nightdata(
        income,
        night
    )

    mai_ics = calculate_mai_ics(
        mai,
        ics
    )

    night_motivation = calculate_nightdata_motivation(
        night,
        motivation
    )

    # --------------------------
    # Final Feature Vector
    # --------------------------

    X = pd.DataFrame([[
        study,
        forum,
        assignment,
        video,
        intention,
        digital,
        motivation,
        barrier,
        mai,
        ics,
        income_night,
        mai_ics,
        night_motivation
    ]],
    columns=[
        "StudyHours_Encoded",
        "Forum_Encoded",
        "Assignments_Encoded",
        "Video_Encoded",
        "DeclaredIntention_Binary",
        "Digital_Literacy_Score",
        "Motivation_Score",
        "Barrier_Score",
        "MAI",
        "ICS",
        "Income_NightData",
        "MAI_ICS",
        "NightData_Motivation"
    ])

    prediction = model.predict(X)[0]

    probability = model.predict_proba(X)[0][1]

    if probability >= 0.70:
        risk = "🔴 High Risk"

    elif probability >= 0.40:
        risk = "🟡 Moderate Risk"

    else:
        risk = "🟢 Low Risk"

    recommendations = generate_recommendations(
        study,
        forum,
        assignment,
        motivation,
        barrier,
        ics
    )

    return (
        risk,
        f"{probability:.1%}",
        recommendations
    )


# ==========================================
# GRADIO INTERFACE
# ==========================================

CSS = """
body{
    background:#f4f7fb;
}
.gradio-container{
    max-width:1400px !important;
    margin:auto;
}
.header{
    background:#003366;
    color:white;
    padding:25px;
    border-radius:18px;
    margin-bottom:20px;
    box-shadow:0px 6px 18px rgba(0,0,0,.15);
}
.title{
    font-size:34px;
    font-weight:bold;
    text-align:center;
}
.subtitle{
    text-align:center;
    font-size:18px;
    opacity:.9;
}
.section{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 12px rgba(0,0,0,.08);
}
.footer{
    text-align:center;
    margin-top:20px;
    color:#666;
    font-size:14px;
}
"""

with gr.Blocks(
    title="FUOYE MOOC Prediction"
) as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Image(
                FUOYE_LOGO,
                show_label=False,
                height=120
            )
        with gr.Column(scale=4):
            gr.HTML("""
            <div class="header">
            <div class="title">
            FUOYE MOOC Engagement & Completion Prediction System
            </div>
            <div class="subtitle">
            Machine Learning & Explainable Artificial Intelligence (XAI)
            </div>
            </div>
            """)
        with gr.Column(scale=1):
            gr.Image(
                EA_LOGO,
                show_label=False,
                height=120
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown("## 📚 Academic Engagement")
            study = gr.Dropdown(
                choices=[
                    "0-2 hours",
                    "3-5 hours",
                    "6-8 hours",
                    "9+ hours"
                ],
                label="Weekly Study Hours"
            )
            forum = gr.Dropdown(
                choices=[
                    "Never",
                    "Rarely",
                    "Sometimes",
                    "Often",
                    "Very Often"
                ],
                label="Forum Participation"
            )
            assignment = gr.Dropdown(
                choices=[
                    "0-25%",
                    "26-50%",
                    "51-75%",
                    "76-100%"
                ],
                label="Assignment Completion"
            )
            video = gr.Dropdown(
                choices=[
                    "0-25%",
                    "26-50%",
                    "51-75%",
                    "76-100%"
                ],
                label="Video Engagement"
            )
            intention = gr.Dropdown(
                choices=[
                    "Intended to complete and earn certificate",
                    "Intended to audit specific modules without completion",
                    "Unsure"
                ],
                label="Declared Learning Intention"
            )
            motivation = gr.Slider(
                1,
                5,
                value=3,
                step=0.1,
                label="Motivation Score"
            )
            barrier = gr.Slider(
                1,
                5,
                value=3,
                step=0.1,
                label="Barrier Score"
            )
        with gr.Column():
            gr.Markdown("## 🌐 Digital & Infrastructure")
            digital = gr.Slider(
                1,
                5,
                value=3,
                step=0.1,
                label="Digital Literacy Score"
            )
            internet = gr.Dropdown(
                choices=[1,2,3,4,5],
                label="Internet Quality"
            )
            electricity = gr.Dropdown(
                choices=[
                    "Less than 2 hours",
                    "2-4 hours",
                    "5-8 hours",
                    "9-12 hours",
                    "More than 12 hours"
                ],
                label="Electricity Availability"
            )
            affordability = gr.Dropdown(
                choices=[1,2,3,4,5],
                label="Data Affordability"
            )
            income = gr.Dropdown(
                choices=[
                    "Under ₦20,000",
                    "₦20,000 - ₦50,000",
                    "₦51,000 - ₦100,000",
                    "Over ₦100,000"
                ],
                label="Monthly Income"
            )
            night = gr.Dropdown(
                choices=[
                    "No, never",
                    "No, rarely",
                    "Yes, frequently",
                    "Yes, exclusively"
                ],
                label="Night Data Usage"
            )

    predict_btn = gr.Button(
        "Predict Student Dropout Risk",
        variant="primary",
        size="lg"
    )

    risk = gr.Textbox(
        label="Risk Level",
        interactive=False
    )

    probability = gr.Textbox(
        label="Dropout Probability",
        interactive=False
    )

    recommendations = gr.Textbox(
        label="Personalized Recommendations",
        lines=8,
        interactive=False
    )

    predict_btn.click(
        fn=predict_dropout,
        inputs=[
            study,
            forum,
            assignment,
            video,
            intention,
            digital,
            motivation,
            barrier,
            internet,
            electricity,
            affordability,
            income,
            night
        ],
        outputs=[
            risk,
            probability,
            recommendations
        ]
    )

if __name__ == "__main__":
    demo.launch(
        css=CSS,
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
