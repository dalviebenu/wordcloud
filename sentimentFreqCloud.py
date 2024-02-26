import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from matplotlib import cm
import matplotlib
import os


def generate_sentiment_wordcloud(file_name):
    # Get the current working directory
    cwd = os.getcwd()
    file_path = os.path.join(cwd, file_name)

    # Read the text file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Tokenize and identify phrases
    tokens = nltk.word_tokenize(text)

    # Common student concern words (in both English and Dutch)
    english_words = ["problem", "issue", "worry", "difficulty", "trouble", "frustration", "housing", "difficulty",
                     "challenge", "obstacle", "disappointment", "stress", "struggle", "finance", "tuition", "fee",
                     "scholarship"]

    dutch_concern_words = ["probleem", "kwestie", "zorg", "moeilijkheid", "problemen", "frustratie", "huisvesting",
                           "moeite",  "uitdaging", "obstakel", "teleurstelling", "stress", "strijd", "financiën", "collegegeld", "vergoeding",
                            "studiebeurs"]

    stop_words = set(STOPWORDS) | {"I", "is", "my", "was", "a"}

    concern_words = english_words + dutch_concern_words
    # Filter and lowercase phrases for case-insensitive uniqueness
    filtered_phrases = []
    for i in range(len(tokens) - 1):
        if tokens[i] in concern_words:
            phrase = tokens[i - 1] + " " + tokens[i] + " " + tokens[i + 1]
            if phrase.lower() not in [p.lower() for p in filtered_phrases]:
                filtered_phrases.append(phrase)



    # Sentiment Analysis
    sia = SentimentIntensityAnalyzer()
    scored_phrases = []
    for phrase in filtered_phrases:
        score = sia.polarity_scores(phrase)['compound']
        scored_phrases.append((phrase, score))

    # Filter out stop words
    filtered_scored_phrases = [(phrase, score) for phrase, score in scored_phrases
                               if all(word not in stop_words for word in phrase.split())]

    # Generate word cloud
    # Offset sentiment scores to be positive
    min_score = min(score for phrase, score in filtered_scored_phrases)  # Find the most negative score
    offset = -min_score + 1  # Ensure all scores become positive (at least 1)
    offset_scored_phrases = [(phrase, score + offset) for phrase, score in filtered_scored_phrases]

    # Generate wordcloud using offset scores
    print(offset_scored_phrases)
    wordcloud = WordCloud(background_color="white").generate_from_frequencies(dict(offset_scored_phrases))

    # Apply color gradient based on sentiment
    sentiment_colormap = matplotlib.colormaps["coolwarm"]

    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        # Find matching phrase entry
        for phrase, score in scored_phrases:
            if phrase == word:
                # Normalize score to [0, 1] for colormap, offset to counter earlier offset
                min_offset_score = 0  # Most negative sentiment + offset becomes 0
                max_offset_score = 2  # Most positive sentiment + offset becomes 2
                norm_score = (score - min_offset_score) / (max_offset_score - min_offset_score)
                print(f"Phrase: {phrase}, norm_score: {norm_score}")
                color_tuple = tuple(int(channel * 255) for channel in sentiment_colormap(norm_score)[:3])
                return color_tuple
        return (0, 0, 0)  # Default color

    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud.recolor(color_func=color_func), interpolation='bilinear')
    plt.axis("off")
    plt.show()


# get file name from command line
# file_name = input("Enter the filename: ")
file_name = "q18.txt"
generate_sentiment_wordcloud(file_name)
