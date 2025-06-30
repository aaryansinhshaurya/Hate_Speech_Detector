# Hate Speech Detection
A robust NLP project that uses Classical ML, Deep Learning (GRU + GloVe), and Transformers (RoBERTa, Toxic-BERT) to classify text(tweet/comment) as Hateful or Non-Hateful

## Team
- Aaryan Sinh Shaurya (SID: 202201075)  
- Denil Antala (SID: 202201090)
- Jami Sidhava (SID: 202201038)

## Goal
Hate speech is a serious and growing issue on the internet.
Our goal is to built a system that can reliably detect hateful content in text.
To do this, we used different types of models — from basic machine learning to advanced transformer models — and combined their strengths into one powerful system that works well even in real-world situations.

## Tech-Stack & Libraries
Data Handling: pandas, numpy, joblib, pickle
Text Cleaning: nltk, re, html, emoji
Visualization: matplotlib, seaborn, wordcloud
ML Models: sklearn (Logistic Regression, Naive Bayes, SVM)
Deep Learning: TensorFlow, Keras (GRU + GloVe)
Transformers: HuggingFace Transformers (RoBERTa, Toxic-BERT)
Deployment: Hugging Face Spaces
Interface: Flask

## Files in this repo
-Dockerfile → Setup for containerized deployment
-Hate_Speech_Detector_v2.ipynb → Main notebook with code & models
-app.py → Web app backend logic
-hatespeech_detection.pdf → Project report and explanation
-hs_gru.h5 → GRU deep learning model (GloVe)
-hs_logreg.joblib → 	Logistic Regression model
-hs_naivebayes.joblib → 	Naive Bayes model
-hs_svm.joblib → 	SVM model
-index.html → 	Web UI interface (frontend)
-requirements.txt → 	List of dependencies
-tfidf_vectorizer.joblib → Saved TF-IDF vectorizer
-tokenizerpkl_gru.pkl → 	Tokenizer for GRU model

## How it works? 
1) Input text (tweet/comment/statement) 
2) We process it and run through different DL models and Transformers
3) Combine predictions using a weighted ensemble
4) Show the ouput (Hate Speech or Not Hate Speech)

## Model Scroes
| Model                | Type                | Accuracy (%) |
| -------------------- | ------------------- | ------------ |
| Logistic Regression  | Classical ML        | 94.08        |
| Naive Bayes          | Classical ML        | 86.83        |
| SVM                  | Classical ML        | 94.52        |
| GloVe - GRU          | Deep Neural Network | 95.80        |
| RoBERTa / Toxic-BERT | Transformer         | –            |
| Ensemble             | Hybrid              | **Best**     |

## Deployment
This project is live on Hugging Face Spaces using FastAPI
- Try it here: https://aaryan24-hate-speech-detector.hf.space/?text=
- Original Dataset Link: https://www.kaggle.com/datasets/waalbannyantudre/hate-speech-detection-curated-dataset/data
- Cleaned Datset Link: https://www.kaggle.com/datasets/h202201075/hate-speech?select=finalhatefull.csv

## Acknowledgements
-Kaggle
-Hugging Face
-Transformers Library
-GloVe Embeddings

## Demo
![image](https://github.com/user-attachments/assets/96d6bd93-db88-4975-aace-2d59c89db108)
![image](https://github.com/user-attachments/assets/eb25cc0b-6d0e-4336-88a5-684fc98121d4)


## License
MIT License

