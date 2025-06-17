import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import time
from transformers import pipeline

STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'on', 'in', 'with', 'to', 'from', 'by',
    'of', 'for', 'at', 'as', 'is', 'it', 'this', 'that', 'these', 'those', 'i', 'you',
    'he', 'she', 'we', 'they', 'them', 'me', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
    'be', 'am', 'are', 'was', 'were', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did'
}

import string

def looks_like_gibberish(word):
    return (
        len(word) < 2 or
        not word.isalpha() or
        re.fullmatch(r"(.)\1{2,}", word) or
        re.search(r'[aeiou]{3,}', word) or
        re.search(r'[zxcvbnm]{4,}', word)
    )

def is_valid_response(response, cue_word):
    tokens = response.lower().strip().split()
    if not 1 <= len(tokens) <= 3:
        return False
    for token in tokens:
        if token == cue_word.lower() or token in STOPWORDS or looks_like_gibberish(token):
            return False
    return True

def calculate_score(label):
    if label == "POSITIVE": return 2
    if label == "NEGATIVE": return -1
    return 1

def format_cue_word(cue):
    return f"""
    <div style='text-align: center; font-size: 32px; font-weight: bold; color: #f7f8fa; padding: 20px;'>
        {cue}
    </div>
    """

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

classifier = load_model()

os.makedirs("results", exist_ok=True)

with open("data/cue_words.txt", "r") as f:
    cue_words = [line.strip() for line in f.readlines()]

with open("data/sentences.txt", "r") as f:
    sentences = [line.strip() for line in f.readlines()]

if "phase" not in st.session_state:
    st.session_state.user_id = ""
    st.session_state.phase = 0
    st.session_state.step = 0
    st.session_state.score = 0
    st.session_state.used_texts = set()
    st.session_state.responses = []
    st.session_state.start_time = None

st.markdown("""
<style>
body {
    background-color: #f6f9fc;
    color: #333;
}
</style>
""", unsafe_allow_html=True)

if st.session_state.phase == 0:
    st.title("Positive Phrase Intervention")
    st.markdown("""
    Welcome to this two-phase task designed to encourage positive associations and emotional reflection.

    - **Phase 1**: Respond to single cue words with uplifting phrases.
    - **Phase 2**: React to full sentences with encouraging responses.
    - Avoid repeats and generic prepositions.
    """)
    user_input = st.text_input("Enter your Name or Roll Number:")
    if st.button("Start Task") and user_input.strip():
        st.session_state.user_id = user_input.strip()
        safe_id = re.sub(r'[^\w\-]', '_', user_input.strip())
        filename = f"results/{safe_id}.csv"
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            st.session_state.responses = df.to_dict("records")
            st.session_state.used_texts = set(df["response"].dropna().str.lower().tolist())
            st.session_state.score = df["score"].sum()
            st.session_state.step = sum(1 for r in st.session_state.responses if r["phase"] == 1)
            st.session_state.phase = 2 if st.session_state.step >= len(cue_words) else 1
        else:
            st.session_state.phase = 1
        st.rerun()

if st.session_state.phase == 1:
    st.progress(st.session_state.step / len(cue_words))
    st.markdown(f"**Points**: `{st.session_state.score}` | **Responses**: `{len(st.session_state.used_texts)}`")

    if st.session_state.step < len(cue_words):
        cue = cue_words[st.session_state.step]
        st.markdown(format_cue_word(cue), unsafe_allow_html=True)

        if st.session_state.start_time is None:
            st.session_state.start_time = time.time()

        feedback = st.empty()

        def handle_input():
            phrase = st.session_state[f"input_{st.session_state.step}"].strip().lower()
            response_time = round(time.time() - st.session_state.start_time, 2)
            result = classifier(phrase)[0]
            label, conf = result['label'], result['score']
            score = calculate_score(label)

            entry = {
                "user": st.session_state.user_id,
                "phase": 1,
                "cue": cue,
                "response": phrase,
                "sentiment": label,
                "confidence": conf,
                "score": 0,
                "response_time_sec": response_time,
                "accepted": False
            }

            if not is_valid_response(phrase, cue):
                feedback.warning("❌ Invalid input: no cue word, gibberish, or stopwords.")
                time.sleep(2)
            elif phrase in st.session_state.used_texts:
                feedback.warning("⚠️ You've already used this. Try something new.")
                time.sleep(2)
            elif label == "NEGATIVE":
                feedback.error("❌ That sounds negative. Try again.")
                time.sleep(2)
            else:
                entry["score"] = score
                entry["accepted"] = True
                st.session_state.score += score
                st.session_state.used_texts.add(phrase)
                st.session_state.step += 1
                st.session_state.start_time = None
                feedback.success(f"✅ Sentiment: {label} ({conf:.2f}) | Score +{score}")
                time.sleep(2)

            st.session_state.responses.append(entry)
            safe_id = re.sub(r'[^\w\-]', '_', st.session_state.user_id)
            pd.DataFrame(st.session_state.responses).to_csv(f"results/{safe_id}.csv", index=False)
            # Removed st.rerun() here

        st.text_input("Type a related uplifting phrase (up to 3 words):", key=f"input_{st.session_state.step}", on_change=handle_input)

    else:
        st.success("🎉 Phase 1 Complete!")
        if st.button("Proceed to Phase 2"):
            st.session_state.step = 0
            st.session_state.phase = 2
            st.rerun()

elif st.session_state.phase == 2:
    st.progress(st.session_state.step / len(sentences))
    st.markdown(f"**Points**: `{st.session_state.score}` | **Responses**: `{len(st.session_state.used_texts)}`")

    if st.session_state.step < len(sentences):
        sentence = sentences[st.session_state.step]
        st.subheader(f"Sentence {st.session_state.step + 1}:")
        st.write(f"**{sentence}**")

        if st.session_state.start_time is None:
            st.session_state.start_time = time.time()

        feedback = st.empty()

        def handle_input_2():
            phrase = st.session_state[f"input_s2_{st.session_state.step}"].strip().lower()
            response_time = round(time.time() - st.session_state.start_time, 2)
            result = classifier(phrase)[0]
            label, conf = result['label'], result['score']
            score = calculate_score(label)

            entry = {
                "user": st.session_state.user_id,
                "phase": 2,
                "sentence": sentence,
                "response": phrase,
                "sentiment": label,
                "confidence": conf,
                "score": 0,
                "response_time_sec": response_time,
                "accepted": False
            }

            if not is_valid_response(phrase, sentence):
                feedback.warning("❌ Invalid input: no sentence word, gibberish, or stopwords.")
                time.sleep(2)
            elif phrase in st.session_state.used_texts:
                feedback.warning("⚠️ You've already used this. Try something new.")
                time.sleep(2)
            elif label == "NEGATIVE":
                feedback.error("❌ That sounds negative. Try again.")
                time.sleep(2)
            else:
                entry["score"] = score
                entry["accepted"] = True
                st.session_state.score += score
                st.session_state.used_texts.add(phrase)
                st.session_state.step += 1
                st.session_state.start_time = None
                feedback.success(f"✅ Sentiment: {label} ({conf:.2f}) | Score +{score}")
                time.sleep(2)

            st.session_state.responses.append(entry)
            safe_id = re.sub(r'[^\w\-]', '_', st.session_state.user_id)
            pd.DataFrame(st.session_state.responses).to_csv(f"results/{safe_id}.csv", index=False)
            # Removed st.rerun() here

        st.text_input("Respond with a positive phrase:", key=f"input_s2_{st.session_state.step}", on_change=handle_input_2)

    else:
        st.balloons()
        st.success("Congratulations! 🎉 Phase 2 Complete!")
        st.markdown(f"**Final Score:** `{st.session_state.score}`")
        df = pd.DataFrame(st.session_state.responses)
        st.dataframe(df)

        st.download_button("📥 Download Your Results", data=df.to_csv(index=False).encode("utf-8"), file_name=f"{st.session_state.user_id}_results.csv")

        st.subheader("📊 Sentiment Confidence Over Time")
        df["step"] = range(1, len(df) + 1)
        fig, ax = plt.subplots()
        df.plot(x='step', y='confidence', kind='line', ax=ax, color='green', marker='o')
        ax.set_title("Sentiment Confidence")
        ax.set_ylabel("Confidence")
        st.pyplot(fig)

        st.subheader("📈 Score Over Time")
        fig, ax = plt.subplots()
        df["cumulative_score"] = df.groupby("phase")["score"].cumsum()
        df.plot(x='step', y='cumulative_score', kind='line', ax=ax, color='blue', marker='o')
        ax.set_title("Cumulative Score")
        ax.set_ylabel("Cumulative Score")
        st.pyplot(fig)

        st.markdown("Thank you for participating! Your responses will help us understand positive emotional associations better.")
        st.markdown("If you want to restart the task, click the button below.")
        if st.button("🔁 Restart"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
