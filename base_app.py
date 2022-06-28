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
	options = ["Prediction", "Model Introduction","Insights", "Meet the Team"]
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

		st.subheader("Interesting Insights into the sample of tweets coolected")
		
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page
            
		st.subheader("Breakdown of the Raw Twitter Data ")
		st.text('This section explains the raw twitter data used to train the models for this App')
		st.image('resources/imgs/piecharttwitet.jpg', width= 500)
		st.text('The dataset contains 15819 tweets across the four sentiment classes. The figure above shows the distribution of the data across the classes. As can be seen, the pro climate change class contains the bulk of the data.')
		st.image('resources/imgs/lengthoftweet.jpg', width= 500)  
		st.text('The figure above represents the average number of words per sentiment class. As can be seen the pro climate change class contains the longest tweets, however, this class does not  differ from the other classes enough to be a significant identifier.')
      
		st.image('resources/imgs/tags.jpg', width= 500)
		st.text('The figure above represents the number of times a tweet was repeated in the dataset. Approximately 14% of the tweets were retweets.')
        
		st.markdown(open('resources/preprocess.md').read())   
        
	# Building out the "Meet the team" page
	if selection == "Meet the Team":
		st.markdown(open('resources/meettheteam.md').read())
		
	# st.image(Image.open('resources/imgs/EDSA_logo.png'),caption=None, use_column_width=True)
	# st.subheader("we are Explore Data Science Academy students. ZF4 Consulting:grin:")
	# st.info("")
	# st.image(Image.open('resources/imgs/Thabi.jpeg'), caption=None, width=250)
	# st.info("")
	# st.image(Image.open('resources/imgs/sima.jpeg'), caption=None, width=250)
    # st.info("")
	# st.image(Image.open('resources/imgs/mokgadi.jpeg'), caption=None, width=250)
	# st.info("")
	# st.image(Image.open('resources/imgs/palesa.jpg'), caption=None, width=250)


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



