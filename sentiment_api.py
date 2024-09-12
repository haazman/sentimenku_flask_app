from flask import Flask, jsonify, request
from flask_cors import CORS
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from transformers import pipeline, AutoTokenizer
import json

app = Flask(__name__)
CORS(app)

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

stop_words_indo = set(stopwords.words('indonesian'))

factory = StemmerFactory()
stemmer = factory.create_stemmer()

towns = ["semarang", "surabaya", "tegal", "sulsel"]

def preprocess_text(text):

    text = text.lower()

    text = re.sub(r'@\w+', '', text)
    
    indo_punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    
    text = re.sub(r'[{}]'.format(re.escape(indo_punctuation)), '', text)
    
    tokens = word_tokenize(text)
    
    lemmatized_tokens = []
    for token in tokens:
        if token not in stop_words_indo:
            lemma = stemmer.stem(token)
            lemmatized_tokens.append(lemma)
    
    preprocessed_text = ' '.join(lemmatized_tokens)
    
    return preprocessed_text

model_name = "mdhugol/indonesia-bert-sentiment-classification"

tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline("sentiment-analysis", model=model_name, tokenizer=tokenizer)

def find_town(text):
    for town in towns:
        if town in text.lower():
            return town
    return None

def get_sentiment(tweets):
    response_data = []
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    count = len(tweets)

    for tweet in tweets:
        text = tweet['text']
        preprocessed_text = preprocess_text(text)
        result = classifier(preprocessed_text)
        label = "Positive" if result[0]['label'] == "LABEL_0" else "Neutral" if result[0]['label'] == "LABEL_1" else "Negative"

        if label == "Positive":
            positive_count += 1
        elif label == "Negative":
            negative_count += 1
        else:
            neutral_count += 1

        town = find_town(text)

        response_data.append({
            'tweet': text,
            'preprocessed_tweet': preprocessed_text,
            'sentiment': label,
            'score': result[0]['score'],
            'created_at': tweet['created_at'],
            'city': town
        })

    return positive_count, negative_count, neutral_count, response_data, count

@app.route('/analyze_tweets', methods=['GET'])
def analyze_tweets():
    with open('sample_twitter_telkomsel-filter.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    tweets = data['data']

    positive_count, negative_count, neutral_count, response_data, count = get_sentiment(tweets)

    positive_percentage = (positive_count / count) * 100 if count > 0 else 0

    response = {
        'tweets': response_data,
        'positive_percentage': positive_percentage,
        'positive_count': positive_count,
        'negative_count': negative_count,
        'neutral_count': neutral_count,
        'total': count,
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
