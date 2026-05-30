# 📰 Fake News Detection System

## 🌐 Live Demo

🔗 https://fake-news-detection-system-jnsbafs2awid7zfmrvxk65.streamlit.app/

---

## 📌 Overview

Fake News Detection System is a Machine Learning based web application that classifies news articles as **Fake News** or **Real News**.

The project uses Natural Language Processing (NLP) techniques and a Logistic Regression model to analyze news content and provide real-time predictions through an interactive Streamlit web interface.

---

## ✨ Features

* Fake News Detection
* Real-Time News Classification
* Interactive Streamlit Web Application
* TF-IDF Text Vectorization
* Logistic Regression Model
* Cloud Deployment using Streamlit
* GitHub Version Control

---

## 🛠 Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-Learn
* Streamlit
* Pickle
* Regex

### Machine Learning

* Logistic Regression

### NLP Technique

* TF-IDF Vectorization

---

## 📊 Dataset Information

The model was trained on a combined dataset containing:

| Dataset   | Articles |
| --------- | -------- |
| Fake News | 23,481   |
| Real News | 21,417   |
| Total     | 44,898   |

### Training Data Characteristics

The dataset mainly contains:

* Political News Articles
* Reuters News Articles
* U.S. Politics Related Content
* Current Affairs News

Because the model is primarily trained on political and Reuters-style news articles, predictions are generally more accurate for news content similar to the training data.

Performance may vary for:

* Sports News
* Entertainment News
* Education News
* Technology News
* Local News

---

## ⚙️ Project Workflow

1. Dataset Collection
2. Data Cleaning
3. Text Preprocessing
4. TF-IDF Vectorization
5. Train-Test Split
6. Logistic Regression Training
7. Model Evaluation
8. Model Saving using Pickle
9. Streamlit Web Application Development
10. Cloud Deployment

---

## 📈 Model Performance

| Metric             | Value               |
| ------------------ | ------------------- |
| Algorithm          | Logistic Regression |
| Feature Extraction | TF-IDF              |
| Dataset Size       | 44,898 Articles     |
| Accuracy           | 98.63%              |

---

## 📁 Project Structure

```text
Fake-News-Detection-System
│
├── app.py
├── main.py
├── fake_news_model.pkl
├── vectorizer.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 How to Run Locally

### Clone Repository

```bash
git clone https://github.com/vk980159-dot/Fake-News-Detection-System.git
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## 📸 Screenshots
<img width="1921" height="819" alt="image" src="https://github.com/user-attachments/assets/a45a6f30-ca4b-479a-85c6-bc120b8e1048" />

<img width="1921" height="819" alt="image" src="https://github.com/user-attachments/assets/3a52c305-d160-428c-88e2-d838b2894c53" />

## 🔮 Future Improvements

* Deep Learning Models
* Multi-Model Comparison
* News URL Analysis
* Dashboard Analytics
* Confidence Score Visualization
* API Integration
* Multilingual News Detection

---

## 👨‍💻 Author

**Vivek Kumar**

Machine Learning | Python | NLP | Streamlit

---

## 📝 Conclusion

This project demonstrates the practical application of Machine Learning and Natural Language Processing for detecting fake news. The system successfully classifies news articles using TF-IDF Vectorization and Logistic Regression with an accuracy of 98.63%, while providing an easy-to-use web interface through Streamlit.
