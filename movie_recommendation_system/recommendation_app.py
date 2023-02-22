import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False) #To ignore streamlit warning 
from PIL import Image
import pandas as pd
# !pip install matplotlib
from matplotlib import pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import numpy as np
import time
# !pip install nltk
import rake_nltk
from rake_nltk import Rake
import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings('ignore')


st.title('Content-Based Movies Recommendation System')

img = Image.open('https://github.com/CodexJoe/Projects/blob/main/movie_recommendation_system/Flix_img.png')
st.image(img, use_column_width= True)

st.header('The project will using Netflix listed movies title to recommend other similar movies for a user that searches a particular movie title')

      

def main():
    stages = ['Exploration and Preprocessing' ,'Visualization', 'Feature Selection','Feature Engineering and Modeling',  'Summary']
    option = st.sidebar.selectbox('Select option:', stages)
    
    if option =='Exploration and Preprocessing':
        st.subheader('Exploratory Data Analysis Stage')

        # data = st.file_uploader('Upload Dataset:', type = ['csv', 'txt'])
        # with st.spinner('Wait for it'):
        #     time.sleep(5)

        # if data is not None:
        #     st.success('Data Succesfully Uploaded')
        
        df = pd.read_csv('https://raw.githubusercontent.com/CodexJoe/Projects/main/movie_recommendation_system/netflix_titles.csv')

        st.write('Reading the first 15 rows')
        st.dataframe(df.head(15))

        if st.checkbox('Display Dimension'):
            dim = st.write(df.ndim)
        if st.checkbox('Display shape'):
            st.write(df.shape)
        if st.checkbox('Display Data type'):
            st.write(df.dtypes)
        if st.checkbox('Display Information'):
            st.write(df.info())
        if st.checkbox('Display Columns'):
            st.write(df.columns)
        if st.checkbox('Display unique values in each column'):
            st.write(df.nunique())
        if st.checkbox('Checking for Duplicates'):
            st.write(df.duplicated().sum())
        if st.checkbox('Display Null Values'):
            st.write(df.isnull().sum())
        if st.checkbox('Replacing Null Values'):
            df['rating'] = df['rating'].fillna(df['rating'].mode()[0])
            df['duration'] = df['duration'].fillna(df['duration'].mode()[0])
            df['date_added'] = df['date_added'].fillna(df['date_added'].mode()[0])
            df['country'] = df['country'].fillna(df['country'].mode()[0])
            df['director'] = df['director'].fillna('Unobtainable')
            df['cast'] = df["cast"].fillna('Unobtainable')
            st.write(df.isnull().sum())



    elif option =='Visualization':
        st.subheader('Data Visualization Stage')
        # data = st.file_uploader('Upload Dataset:', type = ['csv', 'txt'])
        # with st.spinner('Wait for it'):
        #     time.sleep(5)
        # if data is not None:
        #     st.success('Data Succesfully Uploaded')
        
        df = pd.read_csv('https://raw.githubusercontent.com/CodexJoe/Projects/main/movie_recommendation_system/netflix_titles.csv')

        df['rating'] = df['rating'].fillna(df['rating'].mode()[0])
        df['duration'] = df['duration'].fillna(df['duration'].mode()[0])
        df['date_added'] = df['date_added'].fillna(df['date_added'].mode()[0])
        df['country'] = df['country'].fillna(df['country'].mode()[0])
        df['director'] = df['director'].fillna('Unobtainable')
        df['cast'] = df["cast"].fillna('Unobtainable')
        st.write('Reading the first 15 rows')
        st.dataframe(df.head(15))
        
        if st.checkbox('Display Pie Chart of Movie Types'):
            chatpie = df['type']
            pieChart = chatpie.value_counts().plot.pie(autopct = '%1.1f%%')
            plt.legend()
            plt.title('Pie Chart of Movie Types', fontsize = 15, fontweight = 'bold')
            st.write(pieChart)
            st.pyplot()
        if st.checkbox('Display Countplot For Movie Types'):
            plt.figure(figsize = (6,4))
            sns.countplot(x = df['type'])
            st.pyplot()
        if st.checkbox('Display Countplot for Moving Rating'):
            plt.figure(figsize =(10,15))
            sns.countplot(x = df['rating'], order = df['rating'].value_counts().index[0:17])               
            ax= plt.xticks(rotation = 90)
            plt.title('RATINGS AND THEIR COUNTS', fontsize = 18, fontweight = 'bold')
            st.pyplot()
        if st.checkbox('Display Countries with Highest Production'):
            #Visualizing the top 25 movie producing countries 
            plt.figure(figsize = (25,7))
            sns.countplot(x = df['country'], order = df['country'].value_counts().index[0:25], palette="dark")
            ax = plt.xticks(rotation = 90)
            plt.title('25 COUNTRIES WITH HIGHEST PRODUCTION', fontsize = 18, fontweight = 'bold')
            st.pyplot()
        if st.checkbox('Display Countplot of Production Year'):
                # Visualizing the top 20 release year
            plt.figure(figsize = (25,7))
            sns.countplot(x = df['release_year'], order = df['release_year'].value_counts().index[0:21], palette="dark")
            ax = plt.xticks(rotation = 90)
            plt.title('20 TOP MOST MOVIE PRODUCTION YEAR', fontsize = 18, fontweight = 'bold')
            st.pyplot()
        if st.checkbox('Display Countplot of Movie Categories'):        
            #Visualizing the Category of Movie
            plt.figure(figsize = (25,7))
            sns.countplot(x = df['listed_in'].str.split(',').explode(), 
                              order = df['listed_in'].str.split(',').explode().value_counts().index[0:25], palette="dark")
            ax = plt.xticks(rotation = 90)
            plt.title('25 TOP CATEGORIES', fontsize = 18, fontweight = 'bold')
            st.pyplot()
        if st.checkbox('Months contents were added using Wordcloud'):
            months = pd.to_datetime(df['date_added']).dt.month_name() # This formats the date column elements to yy-m-d then extracts the months
            plt.figure(figsize=(9, 8))
            wordcloud = WordCloud(width=900, height=800, background_color='white').generate_from_frequencies(months.value_counts())

            plt.imshow(wordcloud)
            plt.axis('off')
            plt.title('Add Frequency per Month', fontsize = 18, fontweight = 'bold')
            st.pyplot()


    elif option =='Feature Selection':
        # st.subheader('Feature Selection')
        # data = st.file_uploader('Upload Dataset:', type = ['csv', 'txt'])
        # with st.spinner('Wait for it'):
        #     time.sleep(5)
        # if data is not None:
        #     st.success('Data Succesfully Uploaded')
        
        df = pd.read_csv('https://raw.githubusercontent.com/CodexJoe/Projects/main/movie_recommendation_system/netflix_titles.csv')

        df['rating'] = df['rating'].fillna(df['rating'].mode()[0])
        df['duration'] = df['duration'].fillna(df['duration'].mode()[0])
        df['date_added'] = df['date_added'].fillna(df['date_added'].mode()[0])
        df['country'] = df['country'].fillna(df['country'].mode()[0])
        df['director'] = df['director'].fillna('Unobtainable')
        df['cast'] = df["cast"].fillna('Unobtainable')
        st.write('Reading the first 5 rows')
        st.dataframe(df.head(5))
                          
            
        if st.checkbox('Select Features'):
            selected_columns = st.multiselect('Select columns to be used in Recommendation', df.columns)
            movies = df[selected_columns]
            st.dataframe(movies.head(5))
            st.write('......................................................................')
            st.write('The Shape of the new Dataset is: ')

            st.write(movies.shape)
                
               
    elif option =='Feature Engineering and Modeling':
        st.subheader('Feature Engineering Stage')
        # data = st.file_uploader('Upload Dataset:', type = ['csv', 'txt'])
        # with st.spinner('Wait for it'):
        #     time.sleep(5)
        # if data is not None:
            # st.success('Data Succesfully Uploaded')
            # df = pd.read_csv(data)
        df = pd.read_csv('https://raw.githubusercontent.com/CodexJoe/Projects/main/movie_recommendation_system/netflix_titles.csv')
        df['rating'] = df['rating'].fillna(df['rating'].mode()[0])
        df['duration'] = df['duration'].fillna(df['duration'].mode()[0])
        df['date_added'] = df['date_added'].fillna(df['date_added'].mode()[0])
        df['country'] = df['country'].fillna(df['country'].mode()[0])
        df['director'] = df['director'].fillna('Unobtainable')
        df['cast'] = df["cast"].fillna('Unobtainable')
        movies = df.loc[:, ['title', 'type',  'cast', 'rating', 'listed_in', 'description']]
        st.write('Reading the first 5 rows')
        st.dataframe(movies.head(5))
            
        if st.checkbox('Feature Engineering'):
                # Removing movies wrongly rated from the dataset
            rating_error = ['74 min', '84 min', '66 min']
            movies = movies[~movies['rating'].isin(rating_error)]
                    
                # Removing the commas between actors' full names and extracting only the first three actors
            movies['cast'] = movies['cast'].map(lambda x:x.split(',')[:3])
            movies['listed_in'] = movies['listed_in'].map(lambda x:x.split(','))
                # movies['director'] = movies['director'].map(lambda x:x.split(' '))
            movies['type'] = movies['type'].map(lambda x:x.split(' '))
            movies['rating'] = movies['rating'].map(lambda x:x.split('-'))
                
                #Converting all the features to lowercase and concertinating where necessary
            for index, row in movies.iterrows():
                row['cast'] = [x.lower().replace(' ','') for x in row['cast']] # replace(' ','') is used to remove spaces where items are separated by ,
                row['cast'] = [x.replace('-','') for x in row['cast']] # remove - where items are separated by comma
                row['cast'] = [x.replace('.','') for x in row['cast']]
                row['listed_in'] = [x.lower().replace(' ','') for x in row['listed_in']]
                    # row['director'] = ''.join(row['director']).lower()
                row['type'] = ''.join(row['type']).lower()
                row['rating'] = ''.join(row['rating']).lower()
                row['title'] = (row['title']).lower()
            st.write(movies.head(5))

                # Buliding the algorithm
            st.write('--------------------------------------------------------------')
            st.subheader('Buliding The Algorithm')
            if st.checkbox("Extract key words from Movie Descriptions"):
                movies['key_words'] = '' #creating an extra column to hold the keywords

                for index, row in movies.iterrows():
                    description = row['description'] # getting all the rows in description column

                        # Rack() is used to extract key word, it uses english stopwords from NLTK
                        # and discard all puntuation characters

                    r = Rake() # defining an instance of rack
                    r.extract_keywords_from_text(description) # extracting the words by passing the column that contained the text
                    key_words_degree = r.get_word_degrees() #getting dictionary with key words and their score
                    row['key_words'] = list(key_words_degree.keys()) # putting the key words to the new column

                movies.set_index('title', inplace = True) # setting title as index
                movies.drop(columns = ['description'], inplace = True) # dropping description column since we have keywords column now
                st.write(movies.head(5))

            if st.checkbox('Bag of Words'):
                movies['bag_of_words'] = '' # This will contain all the words in all columns joined together
                columns = movies.columns

                for index, row in movies.iterrows():
    
                    words = ''
   
                    for col in columns:
                            # words = words + ' '.join(row[col])+' '
        
                        if col != 'type' and col != 'rating':
                            words = words + ' '.join(row[col])+' '
                        else:
                            words = words + row[col]+ ' '
            
                    row['bag_of_words'] = words
                st.write(movies.head(5))

            if st.checkbox('Performing Count Vectorization on Bag of Words'):
                    # Dropping all other columns but bag_of_words column
                movies.drop(columns = [col for col in movies.columns if col != 'bag_of_words'], inplace = True)
                cv = CountVectorizer() # Instantiating the countvectorizer object
                cv_matrix = cv.fit_transform(movies['bag_of_words'])
                c = cv_matrix.todense()
                st.write(c)   
                
            if st.checkbox('Movies Simialrity Matrix'):
                c_sim = cosine_similarity(cv_matrix, cv_matrix) # checking similarity amongest itselves
                st.write(c_sim[10]) # show similarity of movie in row 6 to all other movies
                
            st.write('--------------------------------------------------------------')
            st.write('This alogrithm will recommend 15 movies from over 8804 listed movies')
                
            if st.checkbox('Recommend Movies'):
                indices = pd.Series(movies.index) # Creating indices for each movie title
                def recommendation(movie_name):

                       
                    try:
                        ind = indices[indices == movie_name].index[0] # This line gets the name from the user and assigns the corresponding index number to it
    
                    except:
                        st.write('Sorry! Movie not Listed') # executes if the movie name is not found 
    
                    else:
                        c_sim_series = list(enumerate(c_sim[ind])) # Converting the cosine similarity matrix to indexed list
                        c_sim_series = sorted(c_sim_series, key = lambda x:x[1], reverse = True) #sorting corresponding cosine similarity matrix in decending order
                        top_15 = c_sim_series[1:16] # Fatches the top 15 highest cosine similarity

    
                        recommended_movies = [i[0] for i in top_15] # iterating over the index of the top 15 list and storing it as recommended_movies
                                # recommended_movies.append(list(movies.index)[i]) # appending the movie names
                        return df[['title', 'director', 'description']].iloc[recommended_movies]
        
                            
                        
                with st.form('Just_Form'):
                    movie_name = st.selectbox('Select Movie Title', df['title'])
                    submitted = st.form_submit_button("Recommend")
                    if submitted:
                            # st.write(movie_name)
                        st.write(recommendation(movie_name.lower()))
                            # st.snow()

                    # movie_name = (input('Search for Movie: ' )).lower()

                    # st.write(recommendation('movie_name', c_sim = c_sim))
        st.header('Bulit by:')
        st.write('Jonathan Okoro')

    elif option == 'Summary':
        st.markdown('The movie industry keep growing bigger')
        
  

                
                    
                    
                    



                

                
    



        
            






if __name__=='__main__':

    main()  