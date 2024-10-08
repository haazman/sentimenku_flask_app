FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader punkt

COPY . .

EXPOSE 8080

CMD ["gunicorn", "-b", ":8080", "sentiment_api:app"]
