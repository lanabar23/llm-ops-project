from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from pymystem3 import Mystem
from tqdm import tqdm
import pandas as pd
import nltk, re
tqdm.pandas()


from config.constants import PATH_TO_TALKDATA

nltk.download('punkt_tab')
nltk.download('stopwords')

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
noise = stopwords.words('russian') 

m = Mystem()
def lemmatize(text, mystem=m):
    try:
        return "".join(m.lemmatize(text)).strip()
    except:
        return " "


# лемматизация
def process_words_review(text, remove_stopwords=True, stops=noise):
    regex = re.compile("[А-Яа-я]+")
    text = ' '.join(re.findall(regex, text))
    words = text.lower().split()
    lemmatized_words = m.lemmatize(" ".join(words))
    if remove_stopwords:
        words = [w for w in lemmatized_words if w not in stops and len(w)>1]
    return words 

# токенизация
def review_to_sentences(text, remove_stopwords=True, stops=noise, tokenizer=sent_tokenize):    
    print(text)
    raw_sentences = tokenizer(text.strip())
    print(raw_sentences)
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            processed_sentence = process_words_review(raw_sentence, remove_stopwords)
            sentences.append(processed_sentence)
    return sentences

# разметка фраз персонажей
def text_lebeling(path_to_file): # df_text)
    # df_pers_phrase = pd.read_table(df_text, header=None)
    df = pd.read_table(path_to_file, header=None)
    df.rename(columns={0:'source_phrase'}, inplace=True)
    df['phrase'] = df['source_phrase'].map(lambda x: str(x).split('- '))
    df['person_phrase'] = df['phrase'].apply(lambda x: x[0]) 
    df['label'] = df['phrase'].apply(lambda x: x[1])
    df.drop(columns=['source_phrase', 'phrase'], inplace=True)
    return df

def data_preproc(path_to_file):  
    df_phrases = text_lebeling(path_to_file)       
    df_phrases['tknz_lem_stopw_text'] = df_phrases['person_phrase'].progress_apply(review_to_sentences)
    print(df_phrases) 
    df_phrases.to_csv('./data/processed_data/phrases.csv')
    # df_phrases.head()


