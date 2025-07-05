
# Hate Speech Detection

A Machine Learning project that detects hate speech in tweets using **Logistic Regression** and **TF-IDF Vectorization**. It classifies whether a tweet contains hate speech or not.

##  Features
- Text preprocessing and cleaning
- TF-IDF vectorization for feature extraction
- Logistic Regression model with class balancing
- Evaluation using confusion matrix and classification report
- Model and vectorizer saved for deployment

## 📊 Dataset
The dataset contains labeled tweets with a `label` column:
- `0` → Non-hate speech
- `1` → Hate speech

## Project Structure
`

hate-speech-detector/
│
├── dataset/
│   ├── train.csv
│   └── test.csv
│
├── hate\_speech.py
├── hate\_speech\_model.pkl
├── tfidf\_vectorizer.pkl
├── confusion matrix \_hate speech detector.png
├── requirements.txt
└── README.md


## Output Sample

![Confusion Matrix](confusion%20matrix%20_hate%20speech%20detector.png)

## Getting Started

1. Clone the repository:

git clone https://github.com/Auromirajayakumar/Hate-Speech-Detection.git
cd Hate-Speech-Detection


2. Install dependencies:


pip install -r requirements.txt

3. Run the project:


python hate_speech.py


##  Requirements

* Python 3.10+
* pandas
* numpy
* nltk
* scikit-learn
* matplotlib
* seaborn

## License

This project is licensed under the **MIT License**.



### 🔗 Connect with Me

Feel free to check out my [LinkedIn](https://www.linkedin.com/in/auromira-jayakumar-1805aa2a9/) 
