import streamlit as st
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pipeline import predict, load_spam_model, load_toxicity_model

st.set_page_config(
    page_title="YouTube Content Moderator",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Oswald:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body { background-color: #000000; color: white; }
        .stApp { background-color: #000000; }
        .main {
            background-color: #000000;
            border-left: 60px solid #FF0000;
            border-right: 60px solid #FF0000;
            padding: 2rem;
        }
        h1, h2, h3 {
            font-family: 'Oswald', sans-serif;
            text-align: center;
            color: white;
        }
        section[data-testid="stSidebar"] {
            background-color: #1a1a1a;
        }
        .stTextArea textarea {
            background-color: #1a1a1a;
            color: white;
            border-radius: 10px;
            border: 1px solid #333;
            font-size: 1rem;
        }
        .stButton button {
            background-color: #CC0000;
            color: white;
            border-radius: 8px;
            border: none;
            font-family: 'Oswald', sans-serif;
            font-size: 1rem;
            width: 100%;
        }
        .stButton button:hover { background-color: #FF0000; }
        .stSlider label { color: white; }
        .stDownloadButton button {
            background-color: #1a1a1a;
            color: white;
            border: 1px solid #CC0000;
            border-radius: 8px;
            width: 100%;
        }
        .badge-flag {
            background-color: #FF0000;
            color: white;
            padding: 0.5rem 1.5rem;
            border-radius: 20px;
            font-family: 'Oswald', sans-serif;
            font-size: 1.2rem;
            display: inline-block;
        }
        .badge-review {
            background-color: #FF9900;
            color: white;
            padding: 0.5rem 1.5rem;
            border-radius: 20px;
            font-family: 'Oswald', sans-serif;
            font-size: 1.2rem;
            display: inline-block;
        }
        .badge-approve {
            background-color: #00AA44;
            color: white;
            padding: 0.5rem 1.5rem;
            border-radius: 20px;
            font-family: 'Oswald', sans-serif;
            font-size: 1.2rem;
            display: inline-block;
        }
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    spam_model = load_spam_model()
    tokenizer, roberta_model = load_toxicity_model()
    return spam_model, tokenizer, roberta_model

spam_model, tokenizer, roberta_model = load_models()

page = st.sidebar.radio(
    "Navigate",
    ["Comment Checker", "Batch Upload", "Data Overview", "Model Performance", "About"]
)

st.markdown("<h1>YouTube Content Moderator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#aaaaaa; font-family:Oswald;'>A two-layer moderation pipeline — spam filter + multi-label toxicity classifier</p>", unsafe_allow_html=True)
st.markdown("---")

if page == "Comment Checker":
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        comment = st.text_area(
            "Enter a comment:",
            label_visibility="collapsed",
            placeholder="Paste a YouTube comment here...",
            height=150
        )
        threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.45, 0.05,
                              help="Adjust how confident the model needs to be before flagging")

        if st.button("Analyze Comment"):
            if comment.strip():
                with st.spinner("Running through pipeline..."):
                    result = predict(comment, spam_model, tokenizer, roberta_model, threshold)

                st.markdown("---")
                action = result['action']
                if action == 'auto_flag':
                    st.markdown('<div style="text-align:center"><span class="badge-flag">🚫 AUTO FLAG</span></div>', unsafe_allow_html=True)
                elif action == 'human_review':
                    st.markdown('<div style="text-align:center"><span class="badge-review">👁 HUMAN REVIEW</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="text-align:center"><span class="badge-approve">✅ AUTO APPROVE</span></div>', unsafe_allow_html=True)

                st.write("")

                if result['spam']:
                    st.error(f"Flagged as spam — confidence: {result['spam_confidence']:.2%}")
                else:
                    if result['labels']:
                        st.write("**Detected categories:**")
                        for label, score in result['label_scores'].items():
                            st.progress(score, text=f"{label}: {score:.2%}")
                    else:
                        st.success("No toxic content detected above threshold")

                st.caption(f"Processed in {result['processing_time_ms']}ms")
            else:
                st.warning("Please enter a comment first.")

elif page == "Batch Upload":
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("<h3>Batch Comment Analysis</h3>", unsafe_allow_html=True)
        st.write("Upload a CSV with a column named **text**. The full pipeline runs on each row and results can be downloaded.")
        uploaded = st.file_uploader("Upload CSV", type="csv", label_visibility="collapsed")
        threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.45, 0.05)

        if uploaded:
            df = pd.read_csv('https://raw.githubusercontent.com/DavidMembreno/youtube-comment-moderation/main/data/processed/processed_toxicity.csv')
            st.write(f"Loaded {len(df)} rows")
            if 'text' not in df.columns:
                st.error("CSV must contain a column named 'text'")
            else:
                if st.button("Run Pipeline"):
                    results = []
                    bar = st.progress(0, text="Processing...")
                    for i, row in enumerate(df['text']):
                        result = predict(str(row), spam_model, tokenizer, roberta_model, threshold)
                        results.append(result)
                        bar.progress((i + 1) / len(df), text=f"Processing {i+1}/{len(df)}...")

                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df)

                    flagged = results_df[results_df['action'] == 'auto_flag']
                    review = results_df[results_df['action'] == 'human_review']
                    approved = results_df[results_df['action'] == 'auto_approve']

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Auto Flagged", len(flagged))
                    c2.metric("Human Review", len(review))
                    c3.metric("Auto Approved", len(approved))

                    st.download_button(
                        "Download Results CSV",
                        results_df.to_csv(index=False),
                        "moderation_results.csv",
                        "text/csv"
                    )

elif page == "Data Overview":
    df = pd.read_csv('https://raw.githubusercontent.com/DavidMembreno/youtube-comment-moderation/main/data/processed/processed_toxicity.csv')
    label_cols = ['toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    def dark_plot(fig, ax):
        ax.set_facecolor('#111111')
        fig.patch.set_facecolor('#111111')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#444')
        ax.spines['left'].set_color('#444')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    st.markdown("<h3>Label Distribution</h3>", unsafe_allow_html=True)
    label_counts = df[label_cols].sum().sort_values(ascending=False)
    label_pcts = (label_counts / len(df) * 100).round(2)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=label_counts.index, y=label_counts.values, color='#CC0000', ax=ax)
    for i, (count, pct) in enumerate(zip(label_counts.values, label_pcts.values)):
        ax.text(i, count + 100, f'{count}\n({pct}%)', ha='center', fontsize=9, color='white')
    dark_plot(fig, ax)
    ax.set_xlabel('Category', color='white')
    ax.set_ylabel('Count', color='white')
    st.pyplot(fig)

    st.markdown("<h3>Text Length vs Toxicity</h3>", unsafe_allow_html=True)
    df['text_length'] = df['text'].apply(len)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.kdeplot(data=df[df['toxic']==0]['text_length'], ax=ax2, label='Not Toxic', color='white', fill=True, alpha=0.3)
    sns.kdeplot(data=df[df['toxic']==1]['text_length'], ax=ax2, label='Toxic', color='#CC0000', fill=True, alpha=0.4)
    ax2.set_xlim(0, 1000)
    dark_plot(fig2, ax2)
    ax2.legend(facecolor='#1a1a1a', labelcolor='white')
    ax2.set_xlabel('Text Length (chars)', color='white')
    st.pyplot(fig2)

    st.markdown("<h3>Label Co-occurrence Heatmap</h3>", unsafe_allow_html=True)
    cooccurrence = df[label_cols].T.dot(df[label_cols])
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cooccurrence, annot=True, fmt='.0f', cmap='Reds',
        ax=ax3, linewidths=0.5, linecolor='#222',
        annot_kws={'color': 'white'}
    )
    fig3.patch.set_facecolor('#111111')
    ax3.set_facecolor('#111111')
    ax3.tick_params(colors='white')
    plt.xticks(rotation=45, ha='right', color='white')
    plt.yticks(color='white')
    st.pyplot(fig3)

    st.markdown("<h3>Multi-label Overlap</h3>", unsafe_allow_html=True)
    overlap = {}
    for label in label_cols:
        flagged = df[df[label] == 1]
        also_flagged = flagged[label_cols].drop(columns=[label]).sum(axis=1)
        overlap[label] = (also_flagged > 0).mean() * 100
    overlap_df = pd.DataFrame({'Label': list(overlap.keys()), 'Overlap %': list(overlap.values())})
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    sns.barplot(x='Label', y='Overlap %', data=overlap_df, color='#CC0000', ax=ax4)
    dark_plot(fig4, ax4)
    ax4.set_xlabel('Label', color='white')
    ax4.set_ylabel('% Also Flagged for Another Label', color='white')
    ax4.set_ylim(0, 100)
    for p in ax4.patches:
        ax4.annotate(f'{p.get_height():.1f}%',
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', color='white', fontsize=9)
    st.pyplot(fig4)

    st.markdown("<h3>Estimated Arbitration Breakdown</h3>", unsafe_allow_html=True)
    st.caption("Estimated distribution at default threshold (0.45) based on dataset label proportions.")
    total = len(df)
    toxic_count = int(df['toxic'].sum())
    borderline = int(total * 0.08)
    auto_flag = int(toxic_count * 0.65)
    auto_approve = total - auto_flag - borderline
    fig5, ax5 = plt.subplots(figsize=(6, 6))
    sizes = [auto_flag, borderline, auto_approve]
    labels = ['Auto Flag', 'Human Review', 'Auto Approve']
    colors = ['#CC0000', '#FF9900', '#00AA44']
    wedges, texts, autotexts = ax5.pie(
        sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90,
        wedgeprops={'linewidth': 2, 'edgecolor': '#111111'}
    )
    for text in texts:
        text.set_color('white')
        text.set_fontfamily('Oswald')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    fig5.patch.set_facecolor('#111111')
    st.pyplot(fig5)

elif page == "Model Performance":
    st.markdown("<h3>Model Results — Run 4 (Final)</h3>", unsafe_allow_html=True)

    perf_df = pd.DataFrame({
        'Label': ['toxic', 'obscene', 'threat', 'insult', 'identity_hate'],
        'Precision': [0.76, 0.74, 0.41, 0.68, 0.51],
        'Recall': [0.90, 0.90, 0.74, 0.85, 0.66],
        'F1': [0.82, 0.81, 0.52, 0.76, 0.57]
    })
    st.dataframe(perf_df, use_container_width=True)

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Macro F1", "0.70")
    c2.metric("Weighted F1", "0.79")
    c3.metric("Spam Accuracy", "93%")
    c4.metric("Spam ROC-AUC", "0.98")

    st.markdown("<h3>Training History</h3>", unsafe_allow_html=True)
    history_df = pd.DataFrame({
        'Epoch': [1, 2, 3, 4, 5, 6],
        'Training Loss': [0.168, 0.090, 0.095, 0.072, 0.057, 0.034],
        'Validation Loss': [0.064, 0.070, 0.067, 0.085, 0.099, 0.108],
        'F1 Macro': [0.595, 0.648, 0.649, 0.677, 0.697, 0.703]
    })
    fig6, ax6 = plt.subplots(figsize=(10, 4))
    ax6.plot(history_df['Epoch'], history_df['F1 Macro'], color='#CC0000', marker='o', label='F1 Macro')
    ax6.plot(history_df['Epoch'], history_df['Validation Loss'], color='white', marker='s', linestyle='--', label='Val Loss')
    ax6.set_facecolor('#111111')
    fig6.patch.set_facecolor('#111111')
    ax6.tick_params(colors='white')
    ax6.legend(facecolor='#1a1a1a', labelcolor='white')
    ax6.spines['bottom'].set_color('#444')
    ax6.spines['left'].set_color('#444')
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    ax6.set_xlabel('Epoch', color='white')
    st.pyplot(fig6)

elif page == "About":
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("<h3>About</h3>", unsafe_allow_html=True)
        st.write("""
        YouTube Content Moderator is a two-layer comment moderation pipeline. A comment passes 
        through a spam filter first, then a fine-tuned RoBERTa model classifies it across five 
        toxicity categories. A confidence arbitration layer determines the final action.
        """)
        st.markdown("---")
        st.write("**Built by** David Membreno — CLU Computer Science, May 2026")
        st.write("**Stack:** Python, HuggingFace Transformers, Scikit-learn, Streamlit")
        st.write("**Models:** Logistic Regression (spam) + Fine-tuned RoBERTa (toxicity)")
        st.write("**Training:** RTX 5070, 6 epochs, focal loss, class weights")
        st.write("**Data:** Jigsaw Toxic Comments + YouTube Toxicity + YouTube Spam Collection")
        st.write("**Repo:** [github.com/DavidMembreno/youtube-comment-moderation](https://github.com/DavidMembreno/youtube-comment-moderation)")