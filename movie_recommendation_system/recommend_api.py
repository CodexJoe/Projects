import random
import string # to process standard python strings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer

movie_title = ['fifty', 'Isoken', 'This lady called life', 'The royal Hibiscus'
               ,'The wedding party', 'A Naija Christmas'] 

movie_description = [
    'Fifty captures few pivotal days of four women at the pinnacle of their careers. Tola, Elizabeth, Maria and Kate are four friends forced at midlife to take inventory at their personal lives, while juggling career and family against the backdrops of the neighbourhoods of Lagos.',
    "Everyone in the Osayande family worries about Isoken. Although she has what appears to have a perfect life beautiful, successful and surrounded by great family and friends. Isoken is still unmarried at 34 which, in a culture obsessed with marriage, is serious cause for concern." ,
    "Aiye (Bisola Aiyeola) who is a young single mother financially struggling to cope up with the rising cost of living in the modern city of Lagos. She works extremely hard running a modest business that barely supports her to have a decent but not a luxury standard of living.",
    "A disenchanted chef tries to help her parents restore their failing hotel but cooks up feelings for an investor with his sights set on the property.",
    "Dozie's (Banky Wellington) elder brother, Nonso (Enyinna Nwigwe), has continued his romance with Deadre (Daniella Down), Dunni's (Adesua Etomi) bridesmaid. Nonso takes Deadre on a date in Dubai and proposes marriage by accident. After a disastrous traditional engagement ceremony in Lagos, Nonso's family and Deadre's aristocratic British family reluctantly agree to a wedding in Dubai.",
    "A Naija Christmas tells the story of an ageing mother who is distraught because her sons have refused to get married and give her grandchildren. She challenges them by offering the first son to marry her Ikoyi house as his inheritance."
]


sent_tokens = movie_description

lemmer = WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
    
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

def recommend_movie(movie_name):
    sent_tokens.append(movie_name)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()   #sort flat
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        return "There's no movie to recommend"
    else:
        return sent_tokens[idx]