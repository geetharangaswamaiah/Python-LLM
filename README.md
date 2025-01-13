# LLM Python
An LLM simulation project in Python which extracts regulations from a given text file and process the text to extract summary of the regulations. Saves the output in a JSON file.

## Quick Start
1. Python version used is 3.11
2. Install requirements: pip install -r requirements.txt
3. Sample regulations data is available in regulations.txt file
4. Run the program: python extract_regulations.py
5. Output will be saved in extracted_requirements.json file


## LLM Simulation Implementation Details:
- Once the regulation file contents are read and sections are split, remove any special characters, digits and extra spaces from the text.
- Download the stopwords using nltk library. Find the frequency of occurrence of each word in the text after removing the stopwords.
- Find the frequency of the most occurring word in the text.
- Divide number of occurrences of the words by maximum frequency to get the weighted frequencies.
- Calculate the score of each sentence based on the sum of the weighted frequencies of the words in the sentence.
- Considering only sentences with less than 30 words to avoid very long sentences in the summary.
- Find the top 3 sentences with the highest scores to generate the summary.
