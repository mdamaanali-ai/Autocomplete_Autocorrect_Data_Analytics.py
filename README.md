# ⌨️ TypeSmart — Autocomplete & Autocorrect (NLP)

An interactive Streamlit app that builds autocomplete and autocorrect from scratch and compares the approaches with precision, recall and speed benchmarks.


## ✨ Features
- **Corpus loading:** Project Gutenberg books via NLTK, your own `.txt` file, or a tiny built-in sample
- **NLP preprocessing:** tokenisation, lowercasing, punctuation removal, stopword removal, with a step-by-step example
- **Autocomplete:** frequency-based unigram, bigram and trigram models with backoff, tested on 10+ inputs with the top 3 predictions
- **Autocorrect:** three edit-distance approaches compared: custom Levenshtein, Norvig-style edits, and `pyspellchecker`
- **Tested on 25 deliberately misspelled words** (editable in the app)
- **Metrics:** precision and recall for both autocomplete and autocorrect, plus speed
- **Visualisations:** top 20 most frequent words, autocorrect confusion matrix (before vs after), comparison charts
- **Live playground:** type text and see suggestions and corrections instantly
- **Discussion:** limitations compared with production keyboards like Google Gboard

## 🛠️ Tech stack
Python · pandas · NLTK · pyspellchecker · Plotly · Streamlit

## 📂 Corpus
By default the app downloads Project Gutenberg books through NLTK on first run (internet needed). You can also upload any `.txt` file from the sidebar. A larger corpus gives better predictions and corrections.

## 🚀 Run locally
```bash
git clone https://github.com/mdamaanali-ai/FUTURE_DS_05.git
cd FUTURE_DS_05
python -m venv venv
venv\Scripts\activate        # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Project structure
```
├── app.py                   # Streamlit application
├── requirements.txt
├── .streamlit/config.toml   # dark theme
└── README.md
```

## 📊 Results
_Run the app on your corpus and add your own numbers, for example:_

| Task | Model / Algorithm | Metric | Score |
|---|---|---|---|
| Autocomplete (next word) | Trigram + backoff | Recall@3 | – |
| Autocomplete (next word) | Bigram | Recall@3 | – |
| Autocorrect | Levenshtein | Recall | – |
| Autocorrect | Norvig edits | Recall | – |

## 🔍 Key takeaways
- Frequency n-grams are a strong, explainable baseline, but they only see one or two words of context.
- Edit distance fixes most typos, but cannot catch real-word errors such as "their" vs "there".
- Results depend heavily on corpus size and domain.


##LIVE DEMO
<p align="center">
  <a href="https://autocompleteautocorrectdataanalyticspy-kv2v3x9zicb7telppciudd.streamlit.app/" target="_blank">
    <img src="https://img.shields.io/badge/🚀_Explore-Live_Demo-7C3AED?style=for-the-badge&logo=streamlit&logoColor=white" alt="Explore Live Demo"/>
  </a>
</p>

## 👤 Author
**mdamaanali-ai** · [GitHub](https://github.com/mdamaanali-ai)

Built as part of the Future Interns Data Science internship (Level 2, Task 5).
