from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


def get_sentiment(headlines):

    if len(headlines) == 0:
        return 0

    scores = []

    for text in headlines:

        sentiment = analyzer.polarity_scores(text)

        scores.append(sentiment["compound"])

    avg_sentiment = sum(scores) / len(scores)

    return avg_sentiment