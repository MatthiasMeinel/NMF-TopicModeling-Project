# NMF-TopicModeling-Project

# Quora Questions Topic Modeling

## Overview
This project focuses on uncovering the hidden patterns in Quora questions using advanced Natural Language Processing (NLP) techniques. Our goal is to categorize a vast array of questions into distinct topics, providing valuable insights into the types of discussions and inquiries popular on Quora.

## Features
- **Data Loading and Preprocessing:** Utilizing `pandas` to load and preprocess Quora questions data.
- **TF-IDF Vectorization:** Transforming the textual data using TF-IDF (Term Frequency-Inverse Document Frequency) to highlight key terms.
- **Topic Modeling with NMF:** Applying Non-Negative Matrix Factorization (NMF) from `scikit-learn` to decompose the dataset into 20 topics.
- **Topic Association:** Integrating the identified topics back into the dataset, associating each question with a specific topic.

## Installation
To run this project, you need to have Python installed along with the following libraries:
- pandas
- scikit-learn

You can install these packages using pip:
```bash
pip install pandas scikit-learn
```

## Usage
1. **Data Loading:**
   ```python
   import pandas as pd
   df = pd.read_csv('quora_questions.csv')
   ```

2. **TF-IDF Vectorization:**
   ```python
   from sklearn.feature_extraction.text import TfidfVectorizer
   tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
   dtm = tfidf.fit_transform(df['Question'])
   ```

3. **NMF Topic Modeling:**
   ```python
   from sklearn.decomposition import NMF
   nmf_model = NMF(n_components=20, random_state=42)
   nmf_model.fit(dtm)
   ```

4. **Display Top Words in Topics:**
   ```python
   for index, topic in enumerate(nmf_model.components_):
       print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
       print([tfidf.get_feature_names()[i] for i in topic.argsort()[-15:]])
       print('\n')
   ```

5. **Assign Topics to Questions:**
   ```python
   topic_results = nmf_model.transform(dtm)
   df['Topic'] = topic_results.argmax(axis=1)
   ```

6. **View the Data:**
   ```python
   print(df.head())
   ```





## Contact
For any queries or discussions, feel free to open an issue in the repository or reach out via email.

---

Happy Exploring! 🚀🌌
