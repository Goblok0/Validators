'''
This code takes in the output of the Online surveys and extract the relevant information for further statistical analysis.

# The implementation of the readability function of textstat is described in the following Kaggle article
  https://www.kaggle.com/code/maxscheijen/text-mining-dutch-news-articles

# 
  https://www.analyticsvidhya.com/blog/2018/02/natural-language-processing-for-beginners-using-textblob/#:~:text=3.6%20Sentiment%20Analysis&text=Polarity%20is%20float%20which%20lies,objective%20refers%20to%20factual%20information.

'''


import numpy as np
import pandas as pd
import re
import unicodedata
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Informativeness packages
import spacy
nlp = spacy.load('nl_core_news_lg')

# Readability packages
import textstat
textstat.set_lang("nl")
from textblob import TextBlob
from textblob_nl import PatternAnalyzer

from statistics import mean

import unidecode

# load in condition file and create separate dataframes for relevant data
def load_df(condition):

    condition_file = "".join(['D:/VU/Validators/Data/Data ', condition,'.xlsx'])
    print(condition_file)

    exc_file = pd.read_excel(condition_file) 
    # Informativeness DataFrame
    df_inf = pd.DataFrame()
    # Answers Dataframe
    df_answers = pd.DataFrame()
    # Time measurement Dataframe
    df_time = pd.DataFrame()
    # Demographic information DataFrame
    df_demo = pd.DataFrame()
    # Polarity assignment DataFrame
    df_emo = pd.DataFrame()
    # Neutral emotional score DataFrame
    df_val = pd.DataFrame()
    # Positive emotional score DataFrame
    df_valp = pd.DataFrame()
    # Negative emotional score DataFrame
    df_valn = pd.DataFrame()
    # User Experience DataFrame
    df_ux = pd.DataFrame()
    # Previous Chatbot experience DataFrame
    df_ux_chatbot = pd.DataFrame()

    df_inf['psid'] = exc_file['psid']

    # Extract the data for the current condition into their respective DataFrame
    if condition == 'A':
          
        df_answers['Q_inzet']      = exc_file.iloc[:,59]
        df_answers['Q_org']        = exc_file.iloc[:,60]
        df_answers['Q_pers']       = exc_file.iloc[:,61]
        df_answers['Q_intent']     = exc_file.iloc[:,62]
        df_answers['Q_disclosure'] = exc_file.iloc[:,64]

    
        df_time['time1'] = exc_file['Tijd Welkom']
        df_time['time2'] = exc_file['Survey tijd 2 surveyconditie - tot en met instructie']
        df_time['time3'] = exc_file['Survey tijd 3 chat condities - tot en met introductievraag']

        df_emo['e_inzet']      = exc_file.iloc[:,73]
        df_emo['e_org']        = exc_file.iloc[:,78]
        df_emo['e_pers']       = exc_file.iloc[:,83]
        df_emo['e_intent']     = exc_file.iloc[:,88]
        df_emo['e_disclosure'] = exc_file.iloc[:,93]

        df_val['val_inzet']      = exc_file.iloc[:,72]
        df_val['val_org']        = exc_file.iloc[:,77]
        df_val['val_pers']       = exc_file.iloc[:,82]
        df_val['val_intent']     = exc_file.iloc[:,87]
        df_val['val_disclosure'] = exc_file.iloc[:,92]

        df_valp['valp_inzet']      = exc_file.iloc[:,70]
        df_valp['valp_org']        = exc_file.iloc[:,75]
        df_valp['valp_pers']       = exc_file.iloc[:,80]
        df_valp['valp_intent']     = exc_file.iloc[:,85]
        df_valp['valp_disclosure'] = exc_file.iloc[:,90]

        df_valn['valn_inzet']      = exc_file.iloc[:,71]
        df_valn['valn_org']        = exc_file.iloc[:,76]
        df_valn['valn_pers']       = exc_file.iloc[:,81]
        df_valn['valn_intent']     = exc_file.iloc[:,86]
        df_valn['valn_disclosure'] = exc_file.iloc[:,91]

    elif condition == 'B':
        df_answers['Q_inzet']      = exc_file['SF Ik hoorde ... ']
        df_answers['Q_org']        = exc_file['SF Waarom heb... ']
        df_answers['Q_pers']       = exc_file['SF Veel mense... ']
        df_answers['Q_intent']     = exc_file['SF Ik voel de... ']
        df_answers['Q_disclosure'] = exc_file['SF Ik wil gra... ']

        df_time['time1'] = exc_file['Tijd Welkom']
        df_time['time2'] = exc_file['Survey tijd 2 chat condities']
        df_time['time3'] = exc_file['Survey tijd 3 chat condities - tot en met introductievraag']

        df_emo['e_inzet']      = exc_file.iloc[:,81]
        df_emo['e_org']        = exc_file.iloc[:,86]
        df_emo['e_pers']       = exc_file.iloc[:,91]
        df_emo['e_intent']     = exc_file.iloc[:,96]
        df_emo['e_disclosure'] = exc_file.iloc[:,101]

        df_emo['e_inzet']      = exc_file.iloc[:,81]
        df_emo['e_org']        = exc_file.iloc[:,86]
        df_emo['e_pers']       = exc_file.iloc[:,91]
        df_emo['e_intent']     = exc_file.iloc[:,96]
        df_emo['e_disclosure'] = exc_file.iloc[:,101]

        df_val['val_inzet']      = exc_file.iloc[:,80]
        df_val['val_org']        = exc_file.iloc[:,85]
        df_val['val_pers']       = exc_file.iloc[:,90]
        df_val['val_intent']     = exc_file.iloc[:,95]
        df_val['val_disclosure'] = exc_file.iloc[:,100]
        
        df_valp['valp_inzet']      = exc_file.iloc[:,78]
        df_valp['valp_org']        = exc_file.iloc[:,83]
        df_valp['valp_pers']       = exc_file.iloc[:,88]
        df_valp['valp_intent']     = exc_file.iloc[:,93]
        df_valp['valp_disclosure'] = exc_file.iloc[:,98]

        df_valn['valn_inzet']      = exc_file.iloc[:,79]
        df_valn['valn_org']        = exc_file.iloc[:,84]
        df_valn['valn_pers']       = exc_file.iloc[:,89]
        df_valn['valn_intent']     = exc_file.iloc[:,94]
        df_valn['valn_disclosure'] = exc_file.iloc[:,99]

        df_ux['ux_telang'] = exc_file.iloc[:,68]
        df_ux['ux_formulering'] = exc_file.iloc[:,69]
        df_ux['ux_saai'] = exc_file.iloc[:,70]
        df_ux['ux_intuitief'] = exc_file.iloc[:,71]
        df_ux['ux_verwarrend'] = exc_file.iloc[:,72]
        df_ux['ux_nadenken'] = exc_file.iloc[:,73]
        df_ux['ux_verkies'] = exc_file.iloc[:,74]
        df_ux_chatbot['ux_chatbot_ervaring'] = exc_file.iloc[:,75]
        df_ux_chatbot['ux_chatbot_survey'] = exc_file.iloc[:,76]

    
    else:
    
        df_answers['Q_inzet']      = exc_file['SF Ik hoorde ... ']
        df_answers['Q_org']        = exc_file['SF Waarom heb... ']
        df_answers['Q_pers']       = exc_file['SF Veel mense... ']
        df_answers['Q_intent']     = exc_file['SF Ik voel de... ']
        df_answers['Q_disclosure'] = exc_file['SF Ik wil gra... ']

        df_time['time1'] = exc_file['Tijd Welkom']
        df_time['time2'] = exc_file['Survey tijd 2 chat condities']
        df_time['time3'] = exc_file['Survey tijd 3 chat condities - tot en met introductievraag']

        #temp_column = [np.nan] * len(exc_file.iloc[:,0])
        df_emo['e_inzet']      = exc_file['SF Ik hoorde ... overall score']
        df_emo['e_org']        = exc_file['SF Waarom heb... overall score']
        df_emo['e_pers']       = exc_file['SF Veel mense... overall score']
        df_emo['e_intent']     = exc_file['SF Ik voel de... overall score']
        df_emo['e_disclosure'] = exc_file['SF Ik wil gra... overall score']

        df_val['val_inzet']      = exc_file['SF Ik hoorde ... neutral score']
        df_val['val_org']        = exc_file['SF Waarom heb... neutral score']
        df_val['val_pers']       = exc_file['SF Veel mense... neutral score']
        df_val['val_intent']     = exc_file['SF Ik voel de... neutral score']
        df_val['val_disclosure'] = exc_file['SF Ik wil gra... neutral score']

        df_valp['valp_inzet']      = exc_file['SF Ik hoorde ... positive score']
        df_valp['valp_org']        = exc_file['SF Waarom heb... positive score']
        df_valp['valp_pers']       = exc_file['SF Veel mense... positive score']
        df_valp['valp_intent']     = exc_file['SF Ik voel de... positive score']
        df_valp['valp_disclosure'] = exc_file['SF Ik wil gra... positive score']

        df_valn['valn_inzet']      = exc_file['SF Ik hoorde ... negative score']
        df_valn['valn_org']        = exc_file['SF Waarom heb... negative score']
        df_valn['valn_pers']       = exc_file['SF Veel mense... negative score']
        df_valn['valn_intent']     = exc_file['SF Ik voel de... negative score']
        df_valn['valn_disclosure'] = exc_file['SF Ik wil gra... negative score']


        df_ux['ux_telang'] = exc_file.iloc[:,63]
        df_ux['ux_formulering'] = exc_file.iloc[:,64]
        df_ux['ux_saai'] = exc_file.iloc[:,65]
        df_ux['ux_intuitief'] = exc_file.iloc[:,66]
        df_ux['ux_verwarrend'] = exc_file.iloc[:,67]
        df_ux['ux_nadenken'] = exc_file.iloc[:,68]
        df_ux['ux_verkies'] = exc_file.iloc[:,69]
        df_ux_chatbot['ux_chatbot_ervaring'] = exc_file.iloc[:,70]
        df_ux_chatbot['ux_chatbot_survey'] = exc_file.iloc[:,71]
        

    # Calculate subjective time in seconds
    sub_min = exc_file['minuten:We willen graag weten hoe lang het onderzoek tot nu voor jou aanvoelt. Geef hieronder in minuten en seconden aan hoe lang het invullen van het onderzoek tot aan deze vraag voor jou voelt.  ']
    sub_sec = exc_file['seconden:We willen graag weten hoe lang het onderzoek tot nu voor jou aanvoelt. Geef hieronder in minuten en seconden aan hoe lang het invullen van het onderzoek tot aan deze vraag voor jou voelt.  ']
    alt_min = sub_min.apply(lambda x: x*60)
    alt_full = alt_min + sub_sec
    df_time['subj_time'] = alt_full

    # get demographic information
    df_demo['gender']   = exc_file['Wat is je geslacht?']
    df_demo['age']      = exc_file['Wat is je leeftijd? ']
    df_demo['edu']      = exc_file['Wat is je hoogst afgeronde opleiding? ']

    return df_inf, df_answers, df_demo, df_time, df_val,df_valp,df_valn, df_emo, df_ux, df_ux_chatbot

# Calculate the wordcount for each answer per questions per respondent
def get_wordmeans(df_answers):
    def word_count(x):
        count = len(x.split(" "))
        return count
    df_word_count = df_answers.applymap(str).applymap(lambda x: word_count(x))   
    
    return df_word_count  

# Calculate the informativeness    
def get_informativeness(df_answers, TFX_df):
    # Apply answer preprocessing and lemmatization and return the informativeness of the answer
    df_inform = df_answers.applymap(str).applymap(lambda x: lemmatize(x, TFX_df))
    
    return df_inform

# Preprocess and lemmatize the answer and calculate the answer's informativeness
def lemmatize(answer,TFX_df):
    
    # sets the answer to lowercase and strips non alphanumeric characters
    strip_answer = re.sub(r'[^\w\s]', '', answer).lower().split(' ')
    pipe_answer = nlp.pipe(strip_answer)

    lemma_answer = []
    score_words = []
    # Go through the each word(/token) in the answer and assign the corresponding term frequency score
    for doc in pipe_answer:
        for token in doc:
            # Lemmatize word
            lem_word = token.lemma_
            lemma_answer.append(remove_accent(lem_word))
            # Try to find the word's corresponding Term Frequency score, 
            # assign a neutral score of zero if the word is not found
            try:
                score_words.append(list(TFX_df.loc[TFX_df['lemma']==lem_word, 'TFX'])[0])
            except: 
                # (Don't assign nan's, instead of just a word it turns the entire answer into a NaN)
                score_words.append(0)

    inf_score = sum(score_words)
    return inf_score

# Readability: Calculate the Flesch Reading Score of the answer
def get_fleshscore(df_answers):
    # apply the flesch function to each answer in the condition
    flesh_reading_ease = lambda x: textstat.flesch_reading_ease(str(x))
    df_flesh = df_answers.applymap(str).applymap(flesh_reading_ease)
    return df_flesh

# # Encode the valence to binary values
# # This was not used in the thesis, but remains in case future use is desired.
def encode_valence(x):
    if x == 'positive':
        return 1
    elif x == 'negative':
        return -1
    elif (x == 'neutral') or (x == 'mixed'):
        return 0
    elif x is None:
        return np.nan
    elif np.isnan(x):
        return np.nan
    elif type(x) == type(0):
        #dataset B, R45C92
        return np.nan
    else:
        print(x)
        raise Exception('Problem encoding valence score')

# Encodes the UX Likert scale answers to numeric values
def encode_ux_score(x):
    if x == 'Helemaal mee eens':
        return 5
    elif x == 'Mee eens':
        return 4
    elif x == 'Mee oneens':
        return 2
    elif x == 'Helemaal mee oneens':
        return 1
    elif x == 'Neutraal':
        return 3
    else:
        raise Exception('Problem encoding ux score')

# Encodes the chatbot Likert scale to numeric values   
def encode_ux_chatbot_ervaring(x):
    if x == 'Nee':
        return 0
    elif x == 'Ja, 1 tot 2 keer':
        return 1
    elif x == 'Ja, 3 tot 4 keer':
        return 2
    elif x == 'Ja, 5 keer of meer':
        return 3
    elif x == 'Weet ik niet':
        return np.nan
    else:
        raise Exception('Problem encoding chatbot_ervaring')

# Encodes the UX scores to numeric values    
def encode_ux_chatbot_survey(x):
    if x == 'Ja':
        return 1
    elif x == 'Nee':
        return 0
    elif x == 'Weet ik niet':
        return np.nan
    elif np.isnan(x):
        return np.nan
    else:
        raise Exception('Problem encoding chatbot survey')

# Set letters with accents to base form (e.g. ä --> a)
def remove_accent(x):
    x = unidecode.unidecode(x)
    return x

# Remove large objective time outliers
def preproc_time(x):
    if x > 1000:
        return np.nan
    else:
        return x


if __name__ == "__main__":
    
    print('start preprocessing')
    TFX_df = pd.read_pickle('D:/VU/Validators/code/TFX_df.pkl')
    conditions = ['A', 'B', 'C']
        
    for condition in conditions:
        print(condition)
        # Load in dataframes of relevant measures from condition data
        try:
            df_inf, df_answers, df_demo, df_time,df_val,df_valp,df_valn, df_emo, df_ux, df_ux_chatbot = load_df(condition)
        except:
            print("File not found or issue loading file")
            continue

        # Extract and process data into analyzable measures
        print('get mean wordcount')
        df_meanwords = get_wordmeans(df_answers)
        df_meanwords = df_meanwords.set_axis(['wc_inzet', 'wc_org', 'wc_pers', 'wc_intent', 'wc_disclosure'], axis=1)

        print('get informativeness')
        df_inform = get_informativeness(df_answers, TFX_df)
        df_inform = df_inform.set_axis(['inf_inzet', 'inf_org', 'inf_pers', 'inf_intent', 'inf_disclosure'], axis=1)

        print('get flesh score')
        df_flesh = get_fleshscore(df_answers)
        df_flesh = df_flesh.set_axis(['read_inzet', 'read_org', 'read_pers', 'read_intent', 'read_disclosure'], axis=1)

        print('encode valence')
        df_emo = df_emo.applymap(encode_valence)
        
        print('process time')
        df_time = df_time.applymap(preproc_time)
        
        # process UX measures, only for condition B and C
        if condition != "A":
            print('encode UX score')
            df_ux = df_ux.applymap(encode_ux_score)
            print('encode UX chatbot1')
            df_ux_chatbot.iloc[:,0] = df_ux_chatbot.iloc[:,0].apply(encode_ux_chatbot_ervaring)
            print('encode UX chatbot2')
            df_ux_chatbot.iloc[:,1] = df_ux_chatbot.iloc[:,1].apply(encode_ux_chatbot_survey)
            df_ux = pd.concat([df_ux, df_ux_chatbot], axis=1)

        print('Combine dataframes into one large dataframe')
        full_df = pd.concat([df_inf, df_demo, df_time, df_answers, df_meanwords, df_inform, df_flesh,df_val,df_valp,df_valn, df_emo, df_ux], axis=1)


        # excluded respondents with nonsense answers, current numbers represent their index in the excel file
        exclude_A = [16,22,31,42,51,53,55]
        exclude_B = [27,41,44,55]
        exclude_C = [1,2,12, 14,15,16,23, 26,32,34,35, 37, 43, 48, 52, 56, 59, 58]
        if condition == 'A':
            exclude = [x-1 for x in exclude_A]
        elif condition == 'B':
            exclude = [x-1 for x in exclude_B]
        else:
            exclude = [x-1 for x in exclude_C]
        PP_df = full_df.drop(exclude)
        # Save processed condition dataframe to excell
        new_file_name = "".join(['D:/VU/Validators/Data/preproc_data_', condition,'.xlsx'])
        print('save PP data to excel')
        PP_df.to_excel(new_file_name)