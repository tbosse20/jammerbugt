import pandas as pd
import matplotlib.pyplot as plt


def create_histogram(excel_file, column_name, min_frequency, exclude_words):
    # Load the Excel file into a DataFrame
    df = pd.read_excel(excel_file)

    # Get the values from the specified column
    column_values = df[column_name].values.tolist()

    # Combine all the text into a single string
    text = ' '.join(str(value) for value in column_values)

    # Remove specific characters
    characters_to_remove = [".", "(", ")", "bund", "rev", "et", "ribber", "dµkke"]
    for char in characters_to_remove:
        text = text.replace(char, "")

    # Split the text into individual words
    words = text.lower().split()

    # Calculate the word frequencies
    word_counts = pd.Series(words).value_counts()

    # Filter out words with frequency less than min_frequency
    word_counts = word_counts[word_counts >= min_frequency]

    # Exclude specified words
    for word in exclude_words:
        if word in word_counts:
            word_counts.drop(word, inplace=True)
    print(word_counts)

    # Plot the histogram
    plt.figure(figsize=(6, 4))
    word_counts.plot(kind='bar')
    plt.xlabel('Word')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Word Frequencies (min_frequency = {min_frequency}) ChatGPT')
    plt.xticks(rotation=30)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust the layout to prevent labels from going out of bounds
    plt.show()


# Example usage
excel_file = 'marta_video.xlsx'
column_name = 'beskrivels'
min_frequency = 5
exclude_words = ['nan', 'er', 'm', 'for', 'langt', '100', 'i', 'over', 'ca', 'information', 'yderligere',
                 'startpunkt', 'reprµsenterer', 'mod', 'punkt', 'ikke', 'link', 'med', 'pσ', 'store', 'Mange', 'og',
                 'spredte', 'dte', '-', 'inde', 'omrσd', '+', 's°kort', 'bσden', 'stor',
                 'af', 'dette', 'en', 'mange', 'til', 'flot', 'enkelte', 'smσ', 'der', 'op',
                 'substrattype', 'bunden', 'meg', 'lµngere', 'nσede', 'videotransekt', 'se', 'nst',

                 'bund', 'kysten', 'kystklint', 'stranden', 'klintekyst', 'vand', 'stikker']
create_histogram(excel_file, column_name, min_frequency, exclude_words)
