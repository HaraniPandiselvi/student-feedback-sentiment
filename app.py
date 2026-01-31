import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from wordcloud import WordCloud

# Load Sentiment Model
sentiment_model = pipeline("sentiment-analysis")

# -------------------------------
# Feedback History Storage
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------
# Topic Detection Function
# -------------------------------
def detect_topic(text):
    text = text.lower()
    if "teacher" in text or "class" in text:
        return "Teaching"
    elif "hostel" in text or "food" in text:
        return "Hostel"
    elif "exam" in text or "test" in text:
        return "Exams"
    elif "placement" in text or "job" in text:
        return "Placement"
    elif "library" in text or "lab" in text or "facility" in text:
        return "Facilities"
    else:
        return "General"

# -------------------------------
# Auto Suggestion Generator
# -------------------------------
def generate_suggestion(topic):
    suggestions = {
        "Teaching": "Improve teaching methods and classroom interaction.",
        "Hostel": "Improve hostel facilities and food quality.",
        "Exams": "Provide better exam preparation support and schedule planning.",
        "Placement": "Increase placement training and industry partnerships.",
        "Facilities": "Upgrade campus infrastructure and maintenance.",
        "General": "Review student concerns and take necessary action."
    }
    return suggestions.get(topic, "Take necessary improvements.")

# -------------------------------
# App Title
# -------------------------------
st.title("🎓 Student Feedback Sentiment Analytics")
st.write("Hackathon Project by Team Technos")

# -------------------------------
# Single Feedback Analysis
# -------------------------------
st.header("📝 Analyze Single Feedback")

feedback = st.text_area("Enter Student Feedback:")

if st.button("Analyze Sentiment"):
    if feedback.strip() == "":
        st.warning("Please enter feedback text!")
    else:
        result = sentiment_model(feedback)[0]
        label = result["label"]

        # Convert Labels + Color Output
        if label == "POSITIVE":
            sentiment = "Positive"
            st.success(f"Sentiment: {sentiment}")
        elif label == "NEGATIVE":
            sentiment = "Negative"
            st.error(f"Sentiment: {sentiment}")
        else:
            sentiment = "Neutral"
            st.warning(f"Sentiment: {sentiment}")

        # Topic Detection
        topic = detect_topic(feedback)
        st.info(f"📌 Topic Category: {topic}")

        # Suggestion for Negative Feedback
        if sentiment == "Negative":
            suggestion = generate_suggestion(topic)
            st.warning(f"💡 Suggested Action: {suggestion}")
        else:
            suggestion = "No action needed"

        # Save into Feedback History
        st.session_state.history.append({
            "Feedback": feedback,
            "Sentiment": sentiment,
            "Topic": topic,
            "Suggestion": suggestion
        })

# -------------------------------
# Feedback History Section
# -------------------------------
st.header("📌 Feedback History")

if len(st.session_state.history) > 0:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df)

    if st.button("🗑 Clear Feedback History"):
        st.session_state.history = []
        st.success("Feedback history cleared!")

else:
    st.info("No feedback history yet. Analyze feedback to store it here.")

# -------------------------------
# CSV Upload and Bulk Analysis
# -------------------------------
st.header("📂 Upload Feedback CSV File")

uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Feedback Data")
    st.dataframe(df)

    sentiments = []
    topics = []
    suggestions = []

    for text in df["feedback"]:
        result = sentiment_model(text)[0]["label"]

        if result == "POSITIVE":
            sentiment = "Positive"
        elif result == "NEGATIVE":
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        topic = detect_topic(text)

        if sentiment == "Negative":
            suggestion = generate_suggestion(topic)
        else:
            suggestion = "No action needed"

        sentiments.append(sentiment)
        topics.append(topic)
        suggestions.append(suggestion)

    df["Sentiment"] = sentiments
    df["Topic"] = topics
    df["Suggestion"] = suggestions

    st.subheader("📌 Final Feedback Analysis Results")
    st.dataframe(df)

    # -------------------------------
    # Sentiment Dashboard Chart
    # -------------------------------
    st.header("📊 Sentiment Distribution Dashboard")

    sentiment_counts = df["Sentiment"].value_counts()

    fig, ax = plt.subplots()
    ax.pie(sentiment_counts,
           labels=sentiment_counts.index,
           autopct="%1.1f%%")
    st.pyplot(fig)

    # -------------------------------
    # Satisfaction Score
    # -------------------------------
    st.header("🎯 College Satisfaction Score")

    positive = (df["Sentiment"] == "Positive").sum()
    total = len(df)

    score = (positive / total) * 100
    st.metric("Overall Satisfaction", f"{score:.1f}%")

    # -------------------------------
    # Word Cloud of Negative Feedback
    # -------------------------------
    st.header("☁ Word Cloud of Negative Feedback")

    negative_text = " ".join(
        df[df["Sentiment"] == "Negative"]["feedback"]
    )

    if negative_text.strip() != "":
        wc = WordCloud(width=800, height=400).generate(negative_text)

        fig, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.info("No negative feedback found!")

    # -------------------------------
    # Download Report
    # -------------------------------
    st.header("⬇ Download Report")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Full Sentiment Report CSV",
        data=csv,
        file_name="sentiment_report.csv",
        mime="text/csv"
    )