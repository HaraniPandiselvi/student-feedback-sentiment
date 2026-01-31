import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from wordcloud import WordCloud

# -------------------------------
# Load Sentiment Model
# -------------------------------
sentiment_model = pipeline("sentiment-analysis")

# -------------------------------
# Convert POS/NEG into Neutral
# -------------------------------
def get_sentiment(text):
    result = sentiment_model(text)[0]
    label = result["label"]
    score = result["score"]

    # Neutral if confidence is low
    if score < 0.70:
        return "Neutral"

    if label == "POSITIVE":
        return "Positive"
    else:
        return "Negative"

# -------------------------------
# Topic Detection Function
# -------------------------------
def detect_topic(text):
    text = text.lower()

    if "hostel" in text or "food" in text:
        return "Hostel"
    elif "teacher" in text or "teaching" in text or "class" in text:
        return "Teaching"
    elif "exam" in text or "test" in text:
        return "Exams"
    elif "placement" in text or "job" in text:
        return "Placement"
    elif "library" in text or "lab" in text or "classroom" in text:
        return "Facilities"
    else:
        return "General"

# -------------------------------
# Suggestion Generator Function
# -------------------------------
def generate_suggestion(topic):
    suggestions = {
        "Hostel": "Improve hostel food quality and hygiene.",
        "Teaching": "Enhance teaching methods and student interaction.",
        "Exams": "Provide better exam preparation and evaluation.",
        "Placement": "Increase placement training and company tie-ups.",
        "Facilities": "Upgrade infrastructure and cleanliness.",
        "General": "Take student feedback seriously for improvement."
    }
    return suggestions.get(topic, "Improve overall student experience.")

# -------------------------------
# Sentiment Color Function
# -------------------------------
def sentiment_color(sentiment):
    if sentiment == "Positive":
        return "green"
    elif sentiment == "Negative":
        return "red"
    else:
        return "orange"

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="Student Feedback Sentiment Analytics", layout="wide")

st.title("🎓 Student Feedback Sentiment Analytics Dashboard")
st.write("Analyze feedback into **Positive, Negative, Neutral** with Topic + Suggestions")

# -------------------------------
# Feedback History Storage
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# Store CSV Results separately
csv_result_df = None

# =====================================================
# ✍ Manual Feedback Input Section
# =====================================================
st.subheader("✍ Enter Student Feedback")

user_feedback = st.text_input("Type feedback here:")

if st.button("Analyze Feedback"):

    if user_feedback.strip() != "":

        sentiment = get_sentiment(user_feedback)
        topic = detect_topic(user_feedback)

        suggestion = ""
        if sentiment == "Negative":
            suggestion = generate_suggestion(topic)

        # Store in history
        st.session_state.history.append({
            "Feedback": user_feedback,
            "Sentiment": sentiment,
            "Topic": topic,
            "Suggestion": suggestion
        })

        # Display Result
        st.markdown(
            f"### Sentiment: <span style='color:{sentiment_color(sentiment)}'>{sentiment}</span>",
            unsafe_allow_html=True
        )

        st.write("📌 Topic:", topic)

        if suggestion:
            st.warning("💡 Suggestion: " + suggestion)

    else:
        st.error("❌ Please enter feedback text.")

# =====================================================
# 📂 CSV Upload Section
# =====================================================
st.subheader("📂 Upload Feedback CSV File")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # Rename Feedback column if needed
    if "Feedback" in df.columns:
        df.rename(columns={"Feedback": "feedback"}, inplace=True)

    if "feedback" not in df.columns:
        st.error("❌ CSV must contain column 'feedback'")
        st.write("Columns found:", df.columns.tolist())
        st.stop()

    results = []

    for text in df["feedback"]:

        if pd.isna(text):
            continue

        text = str(text)

        sentiment = get_sentiment(text)
        topic = detect_topic(text)

        suggestion = ""
        if sentiment == "Negative":
            suggestion = generate_suggestion(topic)

        results.append({
            "Feedback": text,
            "Sentiment": sentiment,
            "Topic": topic,
            "Suggestion": suggestion
        })

    csv_result_df = pd.DataFrame(results)

    st.success("✅ CSV Feedback Analysis Completed!")
    st.dataframe(csv_result_df)

# =====================================================
# 📊 Dashboard Analytics Section (From CSV)
# =====================================================
st.subheader("📊 Analytics Dashboard")

# Priority: CSV Results First
if csv_result_df is not None:

    st.write("### 📌 Analytics Based on Uploaded CSV")

    # Sentiment Distribution Chart
    st.write("### Sentiment Distribution (CSV Data)")

    sentiment_counts = csv_result_df["Sentiment"].value_counts()

    fig, ax = plt.subplots()
    ax.bar(sentiment_counts.index, sentiment_counts.values)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")

    st.pyplot(fig)

    # WordCloud
    st.write("### Word Cloud (CSV Feedback)")

    all_text = " ".join(csv_result_df["Feedback"])

    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)

    fig2, ax2 = plt.subplots()
    ax2.imshow(wordcloud)
    ax2.axis("off")

    st.pyplot(fig2)

# Else use Manual History
elif len(st.session_state.history) > 0:

    st.write("### 📌 Analytics Based on Manual Feedback History")

    history_df = pd.DataFrame(st.session_state.history)

    sentiment_counts = history_df["Sentiment"].value_counts()

    fig, ax = plt.subplots()
    ax.bar(sentiment_counts.index, sentiment_counts.values)

    st.pyplot(fig)

    all_text = " ".join(history_df["Feedback"])

    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)

    fig2, ax2 = plt.subplots()
    ax2.imshow(wordcloud)
    ax2.axis("off")

    st.pyplot(fig2)

    if st.button("🗑 Clear Feedback History"):
        st.session_state.history = []
        st.success("History Cleared!")

else:
    st.info("Upload CSV or analyze feedback to see dashboard.")

# =====================================================
# Footer
# =====================================================
st.markdown("---")
st.markdown("🚀 Hackathon Project by Team **Technos**")