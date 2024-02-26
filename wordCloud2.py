import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from gensim.models.phrases import Phrases, Phraser
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from collections import defaultdict

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')


def generate_word_cloud_with_phrases(text_file):
    # Read the whole text.
    with open(text_file, 'r', encoding='utf-8') as file:
        text = file.read().lower()

    # Tokenize the text into words
    words = word_tokenize(text)

    # Filter out punctuation and stopwords
    stop_words = set(stopwords.words('english'))
    words_cleaned = [word for word in words if word.isalpha() and word not in stop_words]

    # Train bigram model
    bigram_model = Phrases(words_cleaned, min_count=1, threshold=1)
    bigram_phraser = Phraser(bigram_model)

    # Apply the trained bigram phraser to the words list to get bigrams
    bigrams = list(bigram_phraser[words_cleaned])

    # Train trigram model based on the identified bigrams
    trigram_model = Phrases(bigrams, min_count=1, threshold=1)
    trigram_phraser = Phraser(trigram_model)

    # Apply the trained trigram phraser to the bigrams to get trigrams
    trigrams = list(trigram_phraser[bigrams])

    # Count the frequency of individual words
    word_freq = defaultdict(int)
    for word in words_cleaned:
        word_freq[word] += 1

    # Prepare a dictionary for phrase frequencies, using the frequency of the most common word in the phrase
    phrase_freq_dict = defaultdict(int)
    for phrase in trigrams:
        if '_' in phrase:  # This indicates a phrase
            # Replace underscores with spaces for the word cloud and use the frequency of the most common word in the phrase
            phrase_freq_dict[phrase.replace('_', ' ')] = max(word_freq[word] for word in phrase.split('_'))
        else:
            # Also add the frequency of individual words
            phrase_freq_dict[phrase] = word_freq[phrase]

    # Generate a word cloud image from the frequencies
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=800,
                          height=400).generate_from_frequencies(phrase_freq_dict)

    # Display the generated image
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


# Call the function with your text file's path
# Example: generate_word_cloud_with_phrases('path/to/your/file.txt')
generate_word_cloud_with_phrases('q18.txt')
