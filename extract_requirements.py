import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from textblob import TextBlob
from heapq import nlargest
import re
import json
import sys

def read_regulation(regulation_file='regulations.txt'):
    '''
    Read the regulation file
    :param regulation_file: text file containing the regulations
    :return: the file contents
    '''
    try:
        print('Reading the regulation file...')
        with open(regulation_file, 'r') as file:
            return file.read()
    except FileNotFoundError:
        sys.exit('File not found. Please provide the correct file path.')
    except OSError as os_error:
        sys.exit('Error reading the file. Please check the file permissions. {}'.format(os_error))

def split_sections(regulation_contents):
    '''
    Split the regulation contents into sections based on paragraph breaks
    :param regulation_contents: the contents of the regulation file
    :return: a list of dictionaries containing the section number and the original text
    '''
    if regulation_contents is None:
        sys.exit('No regulation contents found. Please provide the correct file path.')
    sections = regulation_contents.split('\n\n')
    output = []
    print('Splitting the regulation contents into sections...')
    for each_section, i in zip(sections, range(1, len(sections)+1)):
        output.append({'Section number': i, 'Original text': each_section})
    return output

def simulate_llm_summary(text_section):
    '''
    Simulate the LLM summary of a given text section
    :param text_section: a section of text from the regulations file
    :return: summary of the text section and metadata
    '''
    print('Simulating LLM summary for the text section...')
    # remove special characters, digits and extra spaces
    preprocessed_text = re.sub(r'\s+', ' ', text_section)
    preprocessed_text = re.sub('[^a-zA-Z]', ' ', preprocessed_text)

    nltk.download('stopwords')
    stop_words = stopwords.words('english')

    # find the frequency of occurrence of each word
    word_frequencies = {}
    for word in word_tokenize(preprocessed_text):
        if word not in stop_words: # ignore stopwords
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1 # initialize the count as word is encountered for the first time
            else:
                word_frequencies[word] += 1 # increment the count as word is encountered again

    maximum_frequency = max(word_frequencies.values()) # frequency of the most occurring word

    # find weighted frequency by dividing number of occurrences of the words by maximum frequency
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / maximum_frequency)

    # calculate the score of each sentence based on the sum of the weighted frequencies of the words in the sentence
    sentence_scores = {}
    sentence_list = sent_tokenize(text_section)
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30: # consider only sentences with less than 30 words as we don't want very long sentences in summary
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    select_len = 3 # number of sentences to include in the summary
    # select the top 'select_len' sentences with highest scores
    summary_sentences = nlargest(select_len, sentence_scores, key=sentence_scores.get)
    summary = ' '.join([word for word in summary_sentences])

    # calculate sentiment analysis of the text section. If polarity is +1, it means the text is positive and if it's -1, it means the text is negative
    sentiment_analysis = round(TextBlob(text_section).sentiment.polarity, 3)
    metadata = {'Original text length': len(text_section), 'Summary length': len(summary), 'Sentiment Analysis': str(sentiment_analysis) +' (+1 means Positive, -1 means Negative)'}
    return summary, metadata

def write_json(data, filename='extracted_requirements_1.json'):
    '''
    Write the extracted requirements to a JSON file
    :param data: summary and metadata of the extracted requirements
    :param filename: json file to write the data to
    '''
    print('Writing the extracted requirements to JSON file {} ...'.format(filename))
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4, separators=(',', ': '))

if __name__ == '__main__':
    regulation_contents = read_regulation('regulations.txt')
    sections = split_sections(regulation_contents)
    for i, each_section in zip(range(len(sections)), sections):
        section_summary, section_metadata = simulate_llm_summary(each_section['Original text'])
        sections[i]['Summary'] = section_summary
        sections[i]['Metadata'] = section_metadata

    write_json(sections, 'extracted_requirements.json')
    print('done')
