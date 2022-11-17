import demoji
import json
import neattext.functions as nfx
import nltk
from nltk.stem.porter import PorterStemmer
import pandas as pd
import pickle
import re
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from transformers import pipeline

# Functions

def lower_demoji(x):
    x = x.lower()
    x = demoji.replace_with_desc(x)
    return x

def stemming(x):
    p_stemmer = PorterStemmer()
    x = p_stemmer.stem(x)
    return x

# Streamlit UI

st.title("Project Elimi-'Hate' Demo")
st.write('Thank you for visiting this API. Write your post below to check if:')
st.write('1. The negative emotion that your post may contain;')
st.write('2. Your post is potentially disrespectful, insulting, offensive, discriminating, humiliating, hateful or dehumanizing towards others; and')
st.write('3. Your post contains any hate words from https://hatebase.org.')

model = load_model('rnn_model.h5')
token = pickle.load(open('rnn_token','rb'))
hatewords = json.loads(open('hatewords.txt','r').read())

form = st.form(key='sentiment-form')
user_input = form.text_area('Enter your post here:')
submit = form.form_submit_button('Submit')

if submit:
   classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)  
   user_input2 = lower_demoji(user_input)
   result = classifier(user_input2)[0]
   label = result[0]['label']
   score = result[1]['score']
   emotion = max(result,key=lambda x:x["score"])["label"]
   if emotion not in ['neutral', 'joy', 'surprise']:
        st.success(f'1. Your post may contain the following negative emotion: {emotion}.')
        user_input1 = lower_demoji(user_input)
        user_input1 = nfx.remove_puncts(user_input1)
        user_input1 = nfx.remove_html_tags(user_input1)
        user_input1 = nfx.remove_special_characters(user_input1)
        user_input1 = stemming(user_input1)
        user_input1 = nfx.remove_stopwords(user_input1)
        user_input1 = token.texts_to_sequences(user_input1)
        user_input1 = sequence.pad_sequences(user_input1, maxlen=256)
        result = model.predict(user_input1)
        result[(result < 0.5)] = 0
        result[(result >= 0.5)] = 1
        result = pd.DataFrame(result, columns = ['disrespectful', 'insulting', 'offensive', 'discriminating', 'humiliating', 'hateful', 'dehumanizing'])
        attributes = []
        for possible_attribute in result:
            if result[possible_attribute][0]: attributes.append(possible_attribute)
        if attributes: st.success(f'2. Your post has the following negative label(s): {", ".join(attributes)}.')
        else: st.success(f'2. Your post does not have any negative labels.')
   else: 
        st.success(f'1. Your post does not contain negative emotion.')
        st.success(f'2. Your post does not contain any negative labels.')

                
if submit: 
  count = 0
  for i in hatewords: 
    if re.match('(^|(.* ))'+i.lower()+'(( .*)|$)',user_input.lower()):
      st.success(f'3. The following word(s) {i} in your post is potentially {hatewords.get(i).lower()}.')
      st.success(f'Please refer to this link for more information: https://hatebase.org/vocabulary/{i.lower().replace(" ", "-")}')
      count =+ 1
  
  if count == 0: st.success(f'3. There is no hate words detected in your post.')
