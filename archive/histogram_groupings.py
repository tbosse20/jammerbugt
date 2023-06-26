import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def create_histogram_from_description(
        freq_name: str,             # Column title for counting words
        stack_name: str,            # Column title for stacking
        separate_word: str,         # Keyword for separating words in stack
        excel_file: str,            # File to process
        min_frequency: int = 0,     # Minimum occurrence of words
        include_words: list = [],   # Only words to include TODO: Didn't work
        exclude_words: list = [],   # Words to ignorer
        excel_rows: int = None,     # Rows in actual excel file
):
    """
    Plot frequency of words used in "beskrivels" seperated by GOPRO files or not.
    (Made with ChatGPT)
    Note: Tried with inclusive words, but didn't work
    """

    df = pd.read_excel(excel_file) # Load the Excel file into a DataFrame

    # Separate the values into two categories: with "GOPR" and without "GOPR"
    with_gopr = df[df[stack_name].str.contains(separate_word, na=False)][freq_name]
    without_gopr = df[~df[stack_name].str.contains(separate_word, na=False)][freq_name]

    # Check if either category is empty
    if with_gopr.empty or without_gopr.empty:
        print("No values found for one or both categories.")
        return

    # Combine the values into a single string for each category
    text_with_gopr = ' '.join(str(value) for value in with_gopr)
    text_without_gopr = ' '.join(str(value) for value in without_gopr)

    # Remove specific characters
    characters_to_remove = [".", "(", ")", ',', "bund", "rev", "et", "ribber", "dµkke", '"', '@', ':', 'grund', '°']
    for char in characters_to_remove:
        text_with_gopr = text_with_gopr.replace(char, "")
        text_without_gopr = text_without_gopr.replace(char, "")

    # Split the text into individual words for each category
    words_with_gopr = text_with_gopr.lower().split()
    words_without_gopr = text_without_gopr.lower().split()

    # Calculate the word frequencies for each category
    counts_with_gopr = pd.Series(words_with_gopr).value_counts()
    counts_without_gopr = pd.Series(words_without_gopr).value_counts()

    # Filter out words with frequency less than min_frequency for each category
    counts_with_gopr = counts_with_gopr[counts_with_gopr >= min_frequency]
    counts_without_gopr = counts_without_gopr[counts_without_gopr >= min_frequency]

    with_gopr_nan = counts_with_gopr["nan"] if (counts_with_gopr.index == 'nan').any() else 0
    without_gopr_nan = counts_without_gopr["nan"] if (counts_without_gopr.index == 'nan').any() else 0
    exclude_words.append('nan')

    # Count excluded words
    with_gopr_removed = 0
    without_gopr_removed = 0
    # Exclude specified words for each category
    for word in exclude_words:
        if word in counts_with_gopr:
            counts_with_gopr.drop(word, inplace=True)
            with_gopr_removed += 1
        if word in counts_without_gopr:
            counts_without_gopr.drop(word, inplace=True)
            without_gopr_removed += 1

    # Limit minimum frequency of words
    counts_with_gopr = counts_with_gopr[counts_with_gopr >= min_frequency]
    counts_without_gopr = counts_without_gopr[counts_without_gopr >= min_frequency]

    # Print status of name, sum, nan, and removed elements in with and without GoPro
    txt = f'File: {excel_file}' + '\n'
    print(txt, end="")
    def status(name, sum, nan, removed, txt):
        print(f'{name} sum: {sum} nan={nan}, (rm words={removed})')
        txt += f'{name} sum: {sum} nan={nan}, (rm words={removed})' + '\n'
        return sum + nan + removed, txt
    total_with, txt = status('With', counts_with_gopr.sum(), with_gopr_nan, with_gopr_removed, txt)
    total_without, txt = status('Without', counts_without_gopr.sum(), without_gopr_nan, without_gopr_removed, txt)
    print(f'Total sum: {total_with + total_without} / {excel_rows} nan={with_gopr_nan + without_gopr_nan}')
    txt += f'Total sum: {total_with + total_without} / {excel_rows} nan={with_gopr_nan + without_gopr_nan}'

    # Create a DataFrame with the merged series
    df = pd.DataFrame({'GoPro': counts_with_gopr, '~GoPro': counts_without_gopr})

    # Plot the histogram
    df.plot(kind='bar', stacked=True)
    plt.xlabel('Word')
    plt.ylabel('Frequency')
    plt.title('Stacked Bar Histogram of Word Frequencies in ' + excel_file)
    plt.xticks(rotation=30)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust the layout to prevent labels from going out of bounds
    plt.legend()
    plt.grid(color='gray', linestyle='dashed', alpha=0.5)
    plt.show()
    print()

# Words excluded in the description
exclude_words = [
    'er', 'm', 'for', 'langt', '100', 'i', 'over', 'ca', 'information', 'yderligere',
    'startpunkt', 'reprµsenterer', 'mod', 'punkt', 'ikke', 'link', 'med', 'pσ', 'store', 'og',
    'spredte', 'dte', '-', 'inde', 'omrσd', '+', 'skort', 'bσden', 'stor',
    'af', 'dette', 'en', 'mange', 'til', 'flot', 'enkelte', 'smσ', 'der', 'op',
    'substrattype', 'bunden', 'meg', 'lµngere', 'nσede', 'videotransekt', 'se', 'nst',
    'bund', 'kysten', 'kystklint', 'stranden', 'klintekyst', 'vand', 'stikker', 'video', 'ved',
    'lige', '5-55', 'las', 'urent', 'nµrmere', 'fisken', '06/08/14', 'lige', 'omrσde', 'fjord',
    'drivende', 'dybt', 'ses', 'ved', 'bred', 'σlegrµs', 'ogsσ', 'midten', 'ingen', 'linealen', 'omrσder', 'observer',
    'sted', 'ud', 'om', 'rundt', 'akkurat', 'fles', 'ude', 'vejen', 'hele', 'n╪', 'mellemrum', '3-5', '5,5',
    '╪', 'vigs', 'bld', 'dyrefod', '50', 'gσr', 'tange', 'vand', '50-100', 'stopper', 'en', '55',
    'vandoverfladen', 'foto_h152p26', 'vej', 's-sv', 'herfra', 'sammenstd', 'staven', 'stod', '30-50', 'samt',
    'tangen', 'klintekysten', 'bσd', 'pga', 'fortsat', 'endnu', 'off', 'videooptagelse', 'n-n╪', 'under', 'targ',
    'passer', 'tµt', 'imellem', 'plter', 'plvist', 'er', 'denne', 'rammer', '20-30', 'aflange', 'umiddelbart',
    '┼legrµs', 'dybere', 'steder', 'ramte', 'videoen', 'd', 'landtange', 'kald', 'stdte', 'stejle', 'hje', 'klinter',
    'overgang', 'nv', '10', 'her', 'sydpσ', 'men', 'dybde', 'stσr', 'blev', 'ad', 'usikkert', 'nσ', 'mellem',
    'som', 'flere', 'st', 'syd', 'stenen'
]

create_histogram_from_description(
    freq_name='beskrivels',
    stack_name='filnavn',
    separate_word='GOPR',
    excel_file='marta_video.xlsx',
    min_frequency=5,
    # include_words=include_words,
    exclude_words=exclude_words,
    excel_rows=1236
)

print('Note: All images starts with "G-" (indicates GoPro)')
create_histogram_from_description(
    freq_name='beskrivels',
    stack_name='metode',
    separate_word='GoPro',
    excel_file='marta_images.xlsx',
    min_frequency=5,
    # include_words=include_words,
    exclude_words=exclude_words,
    excel_rows=6027
)
