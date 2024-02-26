import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import os

# If you haven't already, install NLTK and VADER:
# nltk.download('punkt')
# nltk.download('vader_lexicon')
# pip install vaderSentiment

def generate_concern_wordcloud(file_name):
    # Get the current working directory
    cwd = os.getcwd()

    # Construct the full file path
    file_path = os.path.join(cwd, file_name)

    # Read the text file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Tokenize and identify phrases
    tokens = nltk.word_tokenize(text)

    # Expanded concern words
    concern_words = ["problem", "issue", "worry", "difficulty", "trouble", "frustration", "housing", "difficulty",
                     "challenge", "obstacle", "disappointment", "stress", "struggle", "finance", "tuition", "fee",
                     "scholarship"]

    dutch_concern_words = ["probleem", "kwestie", "zorg", "moeilijkheid", "problemen", "frustratie", "huisvesting",
                           "moeite",  "uitdaging", "obstakel", "teleurstelling", "stress", "strijd", "financiën", "collegegeld", "vergoeding",
                            "studiebeurs"]

    words = concern_words + dutch_concern_words

    # Filtering with concern context
    filtered_phrases = []
    for i in range(len(tokens) - 1):
        if tokens[i] in words:
            phrase = tokens[i - 1] + " " + tokens[i] + " " + tokens[i + 1]
            # Make lowercase for comparison
            if phrase.lower() not in [p.lower() for p in filtered_phrases]:
                filtered_phrases.append(phrase)

    # Sentiment Analysis
    sia = SentimentIntensityAnalyzer()
    scored_phrases = [(phrase, sia.polarity_scores(phrase)['compound']) for phrase in filtered_phrases]

    # Generate word cloud
    wordcloud = WordCloud(background_color="white").generate_from_frequencies(dict(scored_phrases))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

# Example usage
file_name = "q18.txt"  # Replace with your actual filename
generate_concern_wordcloud(file_name)
