# Music_recommender
PROJECT:
An Interactive Music Recommendation System ¨C by Team151

DESCRIPTION:
This project tackles the challenge of building a music recommender system with an interactive interface. It display a list of 10 recommended tracks based on user input of one seed track. The following items are included within this submitted package:

-         /DOC
	Containing the final report and the final poster.

-         /CODE
	All the code files:

	-         /MAIN
		User Interface and Main Recommendation Engine
		The user interface is displayed via HTML5 webpages, which are built upon Python Flask, a micro web framework runs on the user¡¯s local server. This user interface allows user to input seed track, browse for recommendation, give feedback to recommended tracks, and play any of the recommended tracks real time. Connected to the interface is a set of back end Python programs. Wide-and-Deep algorithms and KNN algorithms are used for recommendation purposes. They are built upon TensorFlow and Scikit-Learn packages respectively.
		The sample dataset are also included for demo purpose. The complete dataset is not included to save size of the submitted file.

	-         ALGORITHMS
		Python Script for Algorithm Development

		/KNN
			A Prototype of KNN algorithm for content-based recommendation.
		/ALS
			A Prototype of ALS algorithm for collaborative filtering recommendation.
		/WND
			A Prototype Wide-and-Deep algorithm for Hybrid recommendation.
		/SPOTIFY_API
			A script of how to retrieve data via Spotify Web API. Usage is described within the script file.

	-         /EVALUATIONS
		Evaluation Programs including Downsampling, R-Precision, Intra-List Similarities and Novelty Calculation. Sample testing output files are also included.

PREREQUISITES:
An environment with the following essential packages installed:
-         Python 3.8
-         Flask 1.1.2
-         TensorFlow 2.4.1
-         Scikit-Learn 0.24.1
-         Pandas 1.2.4
-         Numpy 1.19.2
-         Scipy 1.6.2
-         Chrome or Firefox Web Browser
Optional installation for ALS scripts:
-         Hadoop 2.7
-         PySpark 2.1.0
-         Java 7+
 
INSTALLATION:
ALS scripts required installation reference:
https://medium.com/@GalarnykMichael/install-spark-on-windows-pyspark-4498a5d8d66c
Apart from the above required packages, no further installation is needed.
 
EXECUTION:
	1.       Get Started:
	-         Run Command Line (cmd) on Windows OS.
	-         Enter each of the following commands:
	> cd TARGET_DIRECTORY/CODE/Main
	> set FLASK_APP=flaskwebpage.py
	> set FLASK_DEBUG=1
	> python flaskwebpage.py
	-        Open local domain using Chrome or Firefox Browser: http://127.0.0.1:5000

	2.       Get Initial Recommendation:
	-         In the user interface, Input your preferred ¡°track name¡± and the corresponding ¡°artist name¡±, then click ¡°Start Exploring.¡± Button. A recommendation list of 10 tracks will show up, with a live Spotify player to play any of these tracks.

	3.       Provide Feedback and Refresh Recommendation List:
	-         In front of each recommended tracks, three buttons are available for the user:

		1) A ¡®Play¡¯ icon to play the target track using the Spotify player,
		2) A ¡®Like¡¯ (heart icon) to send positive feedback for the chosen track, and the program will refresh based on that selection, and
		3) A ¡®Dislike¡¯ (heart with slash icon) to send negative feedback for the chosen track, and the program will remove that track from the recommendation list.
	
	4.       Others
		The Users are welcome to explore the other background pages:
		            	Home: Return to the home page to start a new round of recommendation.
		            	About: Displays the project introduction as well as team members.
		            	Data Source: The Data being used in the system and their sources.
		            	Algorithm: A description of back-end algorithms being used, with evaluation metrics.

AUTHORS:
	Team151. 
	Team members: Hansong Lin, Shaoou Chen, Minyi Yang, Na Wu, Wenyi Wang and Jia Wen
