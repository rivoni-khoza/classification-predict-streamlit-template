"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""

# Streamlit dependencies
from matplotlib.pyplot import text
import streamlit as st
import joblib,os
import pickle
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import emoji 
from preprocessingfunt import data_preprocessing
from nltk.stem import WordNetLemmatizer
lemm = WordNetLemmatizer()

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/Pickle Dump/Tfidf_Vectorizer.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

def  output(prediction):
    if prediction == -1:
        st.success('This tweet is from someone who does not beieve in man-made Climate Change')
    elif prediction == 0:
        st.success('This tweet is from someone who is Neutral')
    elif  prediction == 1:
        st.success('This tweet is from someone who beieves in man-made Climate Change')
    else:
        st.success('This tweet is a News Article')
    

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	# st.image(['resources/imgs/cars.jpg', 'resources/imgs/park.jpg'], width= 300)
	st.title("Climate Change Tweet Classifier")
	

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Model Introduction","Insights", "About"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Model Introduction":
		
		st.subheader("Model Introduction")

		# You can read a markdown file from supporting resources folder
		st.markdown(open('resources/modelintroduction.md').read())
        
	if selection == "Insights":
		st.subheader("Man-made climate change ")

		# You can read a markdown file from supporting resources folder
		st.markdown(open('resources/mmcc.md').read())

		st.video("https://www.youtube.com/watch?v=sKDWW9WlPSc")

		st.subheader("Insights into the sample of tweets collected")

		st.markdown(open('resources/intro eda.md').read())
		
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

		st.text('The dataset as shown above contains 15819 tweets across the four sentiment classes')
  
		# st.subheader("Breakdown of the Raw Twitter Data")
		st.markdown(open('resources/Fig Explanation 1.md').read())
		
		st.image('resources/imgs/Sentiment_Data_Distribution.png', width= 500)
		st.markdown(open('resources/Fig Explanation 2.md').read())

		st.text('Now, lets have a look at locations:')
		st.image(['resources/imgs/Pro_Climate_Change_People_Locations_Organizations.png', 'resources/imgs/Anti_Climate_Change_People_Locations_Organizations.png'], width= 500)  
		st.markdown(open('resources/Fig Explanation 3.md').read())

		st.text('Hashtag distribution for the various classes:')
		st.image(['resources/imgs/Pro_Climate_Change_Hashtag_Distribution.png', 'resources/imgs/Anti_Climate_Change_Hashtag_distribution.png', 'resources/imgs/Neutral_Sentiments_Hastafg_Distribution.png', 'resources/imgs/News_Related_Hastag_Distribution.png'], width= 500)
		st.markdown(open('resources/Fig Explanation 4.md').read())

		st.text('WordClouds:')
		st.image(['resources/imgs/Pro_Climate_Change_word_cloud.png', 'resources/imgs/Anti_Climate_Sentiment_Word_Cloud.png', 'resources/imgs/News_Related_Word_Cloud.png'], width= 500)
		st.markdown(open('resources/Fig Explanation 5.md').read())
          
        
	# Building out the "About" page
	if selection == "About":
		st.image(('resources/imgs/EDSA_logo.png'),caption=None, use_column_width=True)
		
		st.markdown(open('resources/HIW.md').read())

		st.markdown(open('resources/meettheteam.md').read())
		

		st.info("Rivoni Khoza - Team Leader")
		st.image(('resources/imgs/Rivo1.png'), caption=None, width=250)
		
		st.info("Immanuel Onwochei")
		st.image(('resources/imgs/Ghandi.png'), caption=None, width=250)
    	
		st.info("Raymond Apenteng")
		st.image(('resources/imgs/Me wlp.png'), caption=None, width=250)
		
		st.info("Kelvin Mwaniki")
		st.image(('resources/imgs/Kelvin_Pic.png'), caption=None, width=250)
		
		st.info("Akinbowale Akin-Taylor")
		st.image(('resources/imgs/Akin.png'), caption=None, width=250)
		
		st.info("Peter Adegbe Otanwa")
		st.image(('resources/imgs/PeterPic.png'), caption=None, width=250)

	# Building out the predication page
	if selection == "Prediction":

		st.image(['resources/imgs/cars.jpg', 'resources/imgs/Split.jpg', 'resources/imgs/park.jpg'], width= 230)

		st.subheader("Sentiment Predictor")

		st.info("Prediction with Machine Learning Models")
        
        
		Listmodels = ['Support Vector Classifier','Stack Classifier','logistic Regression Classifier']
		modelselect = st.selectbox('Choose a Model',Listmodels)

		if modelselect == 'Support Vector Classifier':
    			
			# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Tweet Here")

			if st.button("Classify"):
				# Transforming user input with vectorizer
				# vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/Pickle Dump/Linear_SVC.pkl"),"rb"))
				preprocessedtext = data_preprocessing(tweet_text)
				vect_text = tweet_cv.transform(preprocessedtext).toarray()
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				output(prediction)

		if modelselect == 'Stack Classifier':
    			
			# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Tweet Here")

			if st.button("Classify"):
				# Transforming user input with vectorizer
				# vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/Pickle Dump/Stack_Classifier.pkl"),"rb"))
				preprocessedtext = data_preprocessing(tweet_text)
				vect_text = tweet_cv.transform(preprocessedtext).toarray()
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				output(prediction)


		if modelselect == 'logistic Regression Classifier':
    			
			# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Tweet Here")

			if st.button("Classify"):
				# Transforming user input with vectorizer
				# vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/Pickle Dump/Logistic_Regression.pkl"),"rb"))
				preprocessedtext = data_preprocessing(tweet_text)
				vect_text = tweet_cv.transform(preprocessedtext).toarray()
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				
				output(prediction)


# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()



