# Early-detection-of-Parkinson's-Disease-using-Random-Forest-Algorithm
As a part of the Final Year Project for B.Tech degree, A team of three people, me as the Team Lead, developed a Project aimed to predict Parkinson's Disease in a patient using one of the popular ML algorithm, Random Forest Algorithm, by considering few medical attributes in a web application, which would display the test result as either positive or negative.

Project Title:
Early detection of Parkinson’s Disease using Random Forest Algorithm

Summary:
This project aims to provide early detection of Parkinson's disease using popular machine learning algorithms like the Random Forest algorithm and the prediction of the disease would be based on various features like voice frequency, etc.

Software Requirements: 
- Visual Studio Code (VS Code)
- Python 3
	The following Python libraries should be pre-installed in the system: pandas, matplotlib, scikit-learn, Flask, Flask-SocketIO

File Structure:
- `app.py`: 			Flask application for serving the web interface and handling backend functionalities.
- `RandomForest.py`: 	Contains the code for training the Random Forest classifier and generating predictions.
- `SVMALG.py`: 		Placeholder for Support Vector Machine algorithm implementation.
- `static/results`: 		Directory for storing result files and charts generated during the project execution.
- `Main.html`: 			HTML file for the "Home Page" page, which will appear on the opening of the web application.
- `Prediction.html`:		HTML file for the "Checkup" page in the web interface, where the user can get the report.
- `Training.html`:		HTML file for the "Preprocessing the dataset" page in the web interface, where the metrics graph of the algorithms is obtained.
- 'Parkinsons.csv':		The CSV dataset file(195 records and 22 features) containing the records of patients with and without the disease.

Steps to Execute the Code:
1. Clone the repository to your local machine.
2. Extract the downloaded folder and open it in Visual Studio Code.
3. Open the integrated terminal in Visual Studio Code by navigating to `View` > `Terminal`.
4. Navigate to the project directory in the terminal.
5. Ensure all required Python libraries are installed as mentioned in the "Software Requirements" above.
6. Launch the Flask application by running the following command: `python app.py`
7. Once the Flask application is running, open your localhost in your preferred web browser and visit http://localhost:4000 to access the project interface.
8. Use the navigation menu to explore different functionalities such as training, prediction, and checkup.
9. The Home Page provides an overview of Parkinson's Disease, its symptoms, and causes.
10. The Preprocessing page, accessible from the navbar, allows you to preprocess the dataset (`parkinsons.csv` in the zipped folder) and view graphical representations of metrics obtained by RF and SVM algorithms.
11. The Checkup page directs you to a form where you can input your records to determine whether you have the disease. Upon submitting the form, the result will be displayed as either "Positive" or "Negative".
