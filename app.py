# Core Pkgs
import streamlit as st
import altair as alt
import plotly.express as px

# EDA Pkgs
import pandas as pd
import numpy as np
from datetime import datetime

# Utils
import joblib

pipe_lr = joblib.load(open("emotion_classifier.pkl", "rb"))

# Load Database Pkg
import sqlite3

conn = sqlite3.connect("data.db")


# Fxn
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]


def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results


emotions_emoji_dict = {
    "anger": "😠",
    "disgust": "🤮",
    "fear": "😨😱",
    "happy": "🤗",
    "joy": "😂",
    "neutral": "😐",
    "sad": "😔",
    "sadness": "😔",
    "shame": "😳",
    "surprise": "😮",
}


# Main Application
def main():
    st.title("Emotion Classifier App")

    # Create SQLite connection within the main function
    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    # Create tables if not exist
    create_page_visited_table(c)
    create_emotionclf_table(c)

    add_page_visited_details(c, "Home", datetime.now())
    st.subheader("Classification using text")

    with st.form(key="emotion_clf_form"):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label="Submit")

    if submit_text:
        col1, col2 = st.columns(2)

        # Apply Fxn Here
        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        add_prediction_details(
            c, raw_text, prediction, np.max(probability), datetime.now()
        )

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict[prediction]
            st.write("{}:{}".format(prediction, emoji_icon))
            st.write("Confidence:{}".format(np.max(probability)))

        with col2:
            st.success("Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = (
                alt.Chart(proba_df_clean)
                .mark_bar()
                .encode(x="emotions", y="probability", color="emotions")
            )
            st.altair_chart(fig, use_container_width=True)

    # Commit and close SQLite connection
    conn.commit()
    conn.close()


# Fxn To Track Input & Prediction
def create_page_visited_table(c):
    c.execute(
        "CREATE TABLE IF NOT EXISTS pageTrackTable(pagename TEXT,timeOfvisit TIMESTAMP)"
    )


def add_page_visited_details(c, pagename, time_of_visit):
    c.execute(
        "INSERT INTO pageTrackTable(pagename,timeOfvisit) VALUES(?,?)",
        (pagename, time_of_visit),
    )
    conn.commit()


def create_emotionclf_table(c):
    c.execute(
        "CREATE TABLE IF NOT EXISTS emotionclfTable(rawtext TEXT,prediction TEXT,probability NUMBER,timeOfvisit TIMESTAMP)"
    )


def add_prediction_details(c, rawtext, prediction, probability, time_of_visit):
    c.execute(
        "INSERT INTO emotionclfTable(rawtext,prediction,probability,timeOfvisit) VALUES(?,?,?,?)",
        (rawtext, prediction, probability, time_of_visit),
    )
    conn.commit()


if __name__ == "__main__":
    main()
