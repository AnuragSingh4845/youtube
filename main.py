import torch
import streamlit as st
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from math import pi
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.kl import KLSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from rouge import Rouge
from transformers import T5Tokenizer, T5ForConditionalGeneration
from youtube_transcript_api import YouTubeTranscriptApi

nltk.download('punkt')

reference = "The speaker, not an expert but a concerned citizen, addresses the urgent need to confront the climate crisis. They emphasize the undeniable evidence of climate change and its immediate impacts. Urging decisive action, they call for pricing carbon emissions, ending subsidies for fossil fuel companies, and transitioning to renewable energy. Stressing the human rights aspect, they appeal to global leaders to act with courage and honesty in facing this critical challenge."

def get_youtube_transcript(youtube_video):
    video_id = youtube_video.split("v=")[1]
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    source_material = ""
    for i in transcript:
        source_material += ' ' + i['text']
    return source_material

def RGE(summary, reference):
    rouge = Rouge()
    scores = rouge.get_scores(summary, reference, avg=True)
    return scores

def LHM(source_material):
    LANGUAGE = "english"
    parser = PlaintextParser.from_string(source_material, Tokenizer(LANGUAGE))
    summarizer = LuhnSummarizer()
    testsummary = summarizer(parser.document, sentences_count=3)
    summary = ""
    for sentence in testsummary:
        summary += str(sentence)
    return summary

def LSA(source_material):
    LANGUAGE = "english"
    parser = PlaintextParser.from_string(source_material, Tokenizer(LANGUAGE))
    summarizer = LsaSummarizer()
    testsummary = summarizer(parser.document, sentences_count=4)
    summary = ""
    for sentence in testsummary:
        summary += str(sentence)
    return summary

def KL(source_material):
    LANGUAGE = 'english'
    parser = PlaintextParser.from_string(source_material, Tokenizer(LANGUAGE))
    summarizer = KLSummarizer()
    testsummary = summarizer(parser.document, sentences_count=6)
    summary = ""
    for sentence in testsummary:
        summary += str(sentence)
    return summary

def T5(source_material):
    device = torch.device('cpu')
    summarizer = T5ForConditionalGeneration.from_pretrained('t5-base')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    updated_material = "summarize: " + source_material
    tokenized_material = tokenizer.encode(updated_material, return_tensors="pt").to(device)
    tokenized_summary = summarizer.generate(tokenized_material,
                                            num_beams=5,
                                            no_repeat_ngram_size=2,
                                            min_length=100,
                                            max_length=120,
                                            early_stopping=True)
    summary = tokenizer.decode(tokenized_summary[0], skip_special_tokens=True)
    return summary

def plot_rouge_scores_radar(df):
    categories = list(df.columns)
    labels = list(df.index)
    num_vars = len(categories)

    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, label in enumerate(labels):
        values = df.loc[label].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, label=label)
        ax.fill(angles, values, alpha=0.1)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.legend(labels, loc='upper right', bbox_to_anchor=(1.1, 1.1))
    ax.set_title("Summarization Models ROUGE Scores Comparison", size=15, color='blue', y=1.1)

    st.pyplot(fig)

def main():
    st.title("YouTube Transcript Summarization")
    youtube_link = st.text_input("Enter YouTube Link")
    if youtube_link:
        st.video(youtube_link)
        source_material = get_youtube_transcript(youtube_link)
        st.write("TRANSCRIPT FROM YOUTUBE")
        st.write(source_material)

        lhm = LHM(source_material)
        lsa = LSA(source_material)
        kl = KL(source_material)
        t5 = T5(source_material)

        st.title("OUTPUT BY Luhn's Heuristic Method")
        st.write(lhm)
        st.title("OUTPUT BY Latent Semantic Analysis (LSA)")
        st.write(lsa)
        st.title("OUTPUT BY T5 Transformer")
        st.write(t5)
        st.title("OUTPUT BY Kullback-Leibler Sum")
        st.write(kl)

        # Calculate ROUGE scores
        lhm_rouge = RGE(lhm, reference)
        lsa_rouge = RGE(lsa, reference)
        kl_rouge = RGE(kl, reference)
        t5_rouge = RGE(t5, reference)

        # Prepare data for the table
        data = {
            "ROUGE-1 F1": [lhm_rouge['rouge-1']['f'], lsa_rouge['rouge-1']['f'], kl_rouge['rouge-1']['f'], t5_rouge['rouge-1']['f']],
            "ROUGE-1 Precision": [lhm_rouge['rouge-1']['p'], lsa_rouge['rouge-1']['p'], kl_rouge['rouge-1']['p'], t5_rouge['rouge-1']['p']],
            "ROUGE-1 Recall": [lhm_rouge['rouge-1']['r'], lsa_rouge['rouge-1']['r'], kl_rouge['rouge-1']['r'], t5_rouge['rouge-1']['r']],
            "ROUGE-2 F1": [lhm_rouge['rouge-2']['f'], lsa_rouge['rouge-2']['f'], kl_rouge['rouge-2']['f'], t5_rouge['rouge-2']['f']],
            "ROUGE-2 Precision": [lhm_rouge['rouge-2']['p'], lsa_rouge['rouge-2']['p'], kl_rouge['rouge-2']['p'], t5_rouge['rouge-2']['p']],
            "ROUGE-2 Recall": [lhm_rouge['rouge-2']['r'], lsa_rouge['rouge-2']['r'], kl_rouge['rouge-2']['r'], t5_rouge['rouge-2']['r']],
            "ROUGE-L F1": [lhm_rouge['rouge-l']['f'], lsa_rouge['rouge-l']['f'], kl_rouge['rouge-l']['f'], t5_rouge['rouge-l']['f']],
            "ROUGE-L Precision": [lhm_rouge['rouge-l']['p'], lsa_rouge['rouge-l']['p'], kl_rouge['rouge-l']['p'], t5_rouge['rouge-l']['p']],
            "ROUGE-L Recall": [lhm_rouge['rouge-l']['r'], lsa_rouge['rouge-l']['r'], kl_rouge['rouge-l']['r'], t5_rouge['rouge-l']['r']]
        }

        df = pd.DataFrame(data, index=["Luhn's Heuristic Method", "Latent Semantic Analysis (LSA)", "The Kullback-Leibler Sum", "T5 Transformer"])

        st.title("ROUGE Scores Comparison")
        st.dataframe(df)

        st.title("ROUGE Scores Graphical Comparison")
        plot_rouge_scores_radar(df)

if __name__ == "__main__":
    main()
    