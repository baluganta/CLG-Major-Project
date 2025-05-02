# CLG-Major-Project
CREDIT CARD FRAUD DETECTION SYSTEM - using LSTM &amp; XGBoost


 
1.	INTRODUCTION


1.1	Introduction


Fraud detection is a critical aspect of financial security, helping to prevent unauthorized transactions and fraudulent activities. With the increasing adoption of online banking and digital transactions, fraudulent activities have also risen, posing significant risks to businesses and individuals. Detecting fraud in real-time can help mitigate financial losses and maintain trust in digital payment systems. This project implements a fraud detection system using a simple rule-based approach that evaluates transaction details such as Transaction ID, Age, and Amount to determine if a transaction is fraudulent. By leveraging predefined rules, this system can quickly identify suspicious transactions, providing an effective first layer of fraud prevention.


1.2	Motivation

The increasing adoption of online payments has made credit card fraud one of the most common types of financial fraud. Traditional detection methods are rigid and generate many false positives, causing unnecessary disruptions. Financial losses and damaged consumer trust further highlight the need for advanced solutions. Machine learning, especially deep learning models like LSTMs, offers a promising way to detect complex fraud patterns in real-time.


1.3	Objective

The main goal of this project is to create a fraud detection system that checks transactions based on the user's age and the transaction amount. It will have a simple interface where users can enter transaction details and get instant results. The system will give a probability score showing how likely the transaction is to be fraudulent, helping users decide whether to proceed. By using basic rules for quick and efficient fraud detection, this project lays the groundwork for more advanced systems in the future. It is also designed to grow and improve, making it easy to add machine learning and AI-based methods later on.
 
1.4	Scope


This project aims to detect fraudulent transactions by analyzing user age and transaction amount. It uses a Gradio-based web interface, making it easy to access and use for financial institutions, businesses, and researchers. The system provides a basic probability score to show the risk of fraud, acting as a starting point for real-time fraud detection. It can also be integrated into banking systems, online marketplaces, and other financial platforms where fraud detection is important. In the future, the system can be improved with features like real-time monitoring, IP validation, behavioral analysis, and AI-based models to make it more accurate and effective.

1.5	Project Outline

 
2.	LITERATURE SURVEY

Fraud detection has been a widely researched area in financial security, with numerous methodologies implemented to enhance accuracy and efficiency. Various studies have explored rule- based, statistical, and machine learning-based approaches to detecting fraudulent transactions. The increasing sophistication of fraud techniques has led researchers to continuously develop and refine detection models to improve their reliability and efficiency.

	We have studied the following papers for our learning :-

	Real-Time Credit Card Fraud Detection Using Blockchain and AI (2023)

This study explores the integration of blockchain and artificial intelligence (AI) for fraud detection. Blockchain technology enhances security by ensuring a decentralized and transparent transaction ledger, while AI models improve accuracy. The study demonstrates that deep learning techniques, when combined with blockchain verification, can effectively reduce false positives and improve fraud detection accuracy in real time.

	Hybrid Machine Learning Model for Credit Card Fraud Detection (2022)

Researchers developed a hybrid model combining XGBoost, LSTM, and CNN to enhance fraud detection performance. The model was trained on both public and private datasets, achieving high accuracy and F1-score while maintaining low latency. The results indicated that LSTM models, known for their ability to analyze sequential transaction data, significantly improve fraud detection when integrated with boosting techniques like XGBoost.

	Anomaly Detection for Credit Card Fraud Using Autoencoders (2020)

This study investigated the application of autoencoders and neural networks for fraud detection. By training on a combination of synthetic and real transaction data, the model successfully identified anomalous transaction patterns. The research highlighted the potential of unsupervised learning methods in detecting fraudulent activities without prior labeled data, making them valuable for real-time fraud detection.
 
	Credit Card Fraud Detection Using Random Forest and SVM (2019)

A comparative study between Random Forest and Support Vector Machines (SVM) was conducted using the ULB (European Credit Card Fraud) dataset. The study found that Random Forest outperformed SVM in terms of AUC-ROC, precision, and recall, making it a preferable choice for fraud detection. While effective, traditional ML techniques like Random Forest struggle to detect sequential fraud patterns, making deep learning methods like LSTM more suitable for real-time fraud detection.
 
 
3.	SYSTEM STUDY AND ANALYSIS

3.1	Problem Statement


With the rise in online transactions, fraudulent activities have increased significantly, causing financial losses and security risks. The current fraud detection methods are either too slow or not adaptable to evolving fraud tactics. A system is needed to detect fraud in real-time using efficient and scalable techniques.

3.2	Existing System


The existing fraud detection systems rely on traditional rule-based approaches or manual reviews, which often fail to keep up with modern fraud tactics. Some systems use machine learning models but suffer from issues like high false positives and computational complexity.



3.3	Limitations of Existing system

•	Rule-based systems lack adaptability to new fraud techniques.

•	Manual fraud detection is slow and inefficient.

•	High false-positive rates in some machine learning-based systems.

•	Computationally expensive models may not be feasible for small businesses.




3.4	Proposed System

The proposed system uses a rule-based fraud detection approach that evaluates transactions based on predefined criteria such as age and transaction amount. The system is designed for real-time evaluation and provides a probability score to help users assess transaction risks.
 
3.5	Advantages of Proposed System

•	Real-time fraud detection for instant decision-making.

•	Easy to implement and scalable for future enhancements.

•	Reduces false positives by applying multiple validation criteria

•	Can be integrated into existing financial systems.



3.6	Functional Requirements


	Users can input transaction details.

	The system processes the transaction based on predefined rules.

	Generates a prediction and probability score.

	Provides real-time fraud detection output.


3.7	Non Functional Requirements


	System should be efficient and fast.

	User-friendly interface using Gradio.

	Secure handling of transaction data.

	Scalable for future enhancements
 


3.8	System Requirements
	Hardware: Minimum 4GB RAM, 1GHz processor.

	Software: Python, Gradio, Web Browser.

	Libraries: NumPy, Pandas, Scikit-learn.



3.9	Models Used
	XG Boost
	LSTM

  
4.	About LSTM (Long Short-Term Memory)


Long Short-Term Memory (LSTM) is a special kind of Recurrent Neural Network (RNN) architecture designed specifically to model and understand sequential data, where the order and timing of inputs matter. Traditional RNNs suffer from problems like the vanishing gradient, which makes them ineffective at learning long-term dependencies in sequences. LSTM overcomes this limitation by introducing memory cells and gating mechanism namely the input gate, forget gate, and output gate. These gates help the model decide which information to remember, which to forget, and which to pass on to the next step in the sequence.
In the context of credit card fraud detection, transactions occur as a sequence of events over time. LSTM networks are well-suited to capture temporal patterns and behavioral trends in this sequence. For instance, if a user typically makes small transactions in one city and suddenly makes a large transaction in a foreign country, this shift in behavior can be flagged as suspicious. LSTM can effectively learn such patterns and anomalies by retaining relevant past information over longer periods.
5.1.2	Working of LSTM in Fraud Detection
The process of using LSTM for fraud detection involves several crucial steps:
1.	Data Collection and Preparation
Historical transaction data is collected, usually containing information such as:
•	Transaction ID
•	Transaction amount
•	Timestamp
•	Location
•	Payment method
•	Merchant category
•	User demographic data (e.g., age, credit score, previous fraud history)
2.	Preprocessing and Feature Engineering
Since raw transactional data often contains categorical fields (like "Payment Method" or "Device Type"),  these are converted into numerical formats using encoding techniques like
 
Label Encoding or One-Hot Encoding.
3.	LSTM Network Architecture
The LSTM model typically consists of:
•	Input Layer: Accepts the sequence of transactions.
•	LSTM Layers: These layers are the heart of the model. They analyze transaction sequences while preserving memory over time. They are capable of learning both short-term fluctuations and long- term behavior patterns in transaction history.
•	Dropout Layers: Used between LSTM layers to reduce overfitting by randomly turning off neurons during training.
•	Batch Normalization: Ensures faster convergence and better generalization.
•	Dense Layers: These are fully connected layers that perform final classification. The last layer usually uses a sigmoid activation to output a probability of whether the transaction is fraudulent or not.
4.	Model Training
The model is trained using historical labeled data (fraudulent vs. non-fraudulent transactions). The binary cross-entropy loss function is commonly used for classification tasks, and Adam optimizer is frequently employed for efficient learning.
5.	Integration with Other Models
In advanced systems, the output of LSTM can be used as a feature extractor. The extracted patterns from LSTM can be passed to another machine learning model like XGBoost for final classification. This hybrid model architecture combines the temporal sequence learning ability of LSTM with the powerful classification ability of XGBoost, improving overall accuracy and robustness.
6.	Prediction and Deployment
Once trained, the model is deployed as part of a fraud detection pipeline. During real-time transaction processing, new transaction sequences are passed to the model, which then outputs a fraud probability. Transactions exceeding a certain threshold are flagged for further review or blocked automatically.
 
4.1	Advantages of LSTM

•	Handles Long-Term Dependencies: LSTMs effectively retain information over long sequences, unlike traditional RNNs.
•	Mitigates Vanishing Gradient Problem: The gating mechanism prevents gradients from diminishing, ensuring stable training.
•	Selective Memory Retention: Input, forget, and output gates allow the model to selectively store or discard information.
•	Effective for Sequential Data: Ideal for applications like time series forecasting, NLP, and speech recognition.
•	Robust to Noisy Data: Can learn patterns even in datasets with inconsistencies or missing values.
•	Supports Parallelization: While inherently sequential, modern frameworks enable some level of parallel computation.
 
5	XGBoost(Extreme Gradient Boosting)
	About XGBoost

XGBoost stands for Extreme Gradient Boosting. It is a very fast and accurate machine learning algorithm that is often used in real-world problems like credit card fraud detection, customer churn, medical diagnosis, etc.XGBoost is based on a method called gradient boosting, where the idea is to build several small decision trees (also called weak learners) and combine them to make a strong final prediction. Each new tree tries to fix the mistakes made by the previous ones, step by step.
5.2.2	Working of XGBoost in Fraud Detection

The process of using XGBoost for credit card fraud detection starts with data collection and preprocessing. We begin by gathering historical transaction data, which includes details such as the transaction amount, time, type (like online, POS, or ATM), location, user ID, and whether the transaction was fraudulent or not. Since machine learning models require numerical input, any text or categorical data— such as the type of transaction is converted into numbers using encoding techniques. In addition, all numerical values are scaled or normalized to ensure they fall within a similar range. This step is crucial for helping the model learn efficiently and consistently from the data.
Next comes the challenge of handling imbalanced data. In real-world scenarios, fraudulent transactions are very rare, often making up only a tiny fraction of the overall dataset—sometimes just one in a thousand transactions. This imbalance can lead the model to focus mostly on non-fraud cases, resulting in poor fraud detection. To overcome this, we use techniques like SMOTE (Synthetic Minority Oversampling Technique) to generate more synthetic fraud cases or undersampling to reduce the number of non-fraud cases. These methods help in balancing the dataset, allowing the model to learn patterns from both fraud and non- fraud transactions more effectively.After preparing the data, we proceed to the splitting and training phase. The dataset is divided into two parts: a training set and a testing set. The training set is used to teach the model by feeding it transaction features (like amount, time, and type) along with the corresponding fraud labels. The testing set helps evaluate how well the model performs on new, unseen data. While building the
 
XGBoost model, we also adjust important parameters such as the learning rate (which controls how quickly the model learns), tree depth (which defines how complex each decision tree is), and the number of estimators (which is the number of trees in the model). Tuning these parameters improves the model's performance and reduces prediction errors.Finally, after training, the model is ready for making predictions. When a new transaction is input into the system, the trained XGBoost model analyzes the details and gives a probability score that indicates the likelihood of fraud. If the score is higher than a predefined threshold, the transaction is flagged as fraudulent. This approach allows the system to detect suspicious activity with high accuracy, helping financial institutions respond quickly and reduce the risk of financial loss.

5.1	Advantages of XGBoost


•	High Accuracy: XGBoost effectively learns complex fraud patterns and reduces false positives, ensuring reliable fraud detection.

•	Handles Imbalanced Data: Since fraud cases are rare, XGBoost manages imbalanced datasets well using techniques like oversampling (SMOTE).

•	Fast Processing Speed: It quickly analyzes large transaction datasets, making it ideal for real-time fraud detection.

•	Prevents Overfitting: Built-in regularization (L1& L2) helps the model generalize better, preventing it from memorizing noise in the data.

•	Feature Importance & Interpretability: XGBoost shows which transaction features (amount, location, or time) contribute most to fraud detection, helping improve security strategies.
 
5.2.4	Model Architecture(XGBoost)



 
5.3	Python

5.3.1	About Python

Python is a high-level, versatile, and interpreted programming language known for its simplicity, readability, and widespread adoption. Created by Guido van Rossum in the late 1980s, Python has become one of the most popular languages for a wide range of applications.
Python's readability is a key strength, as its syntax emphasizes code clarity through the use of whitespace and indentation, making it an excellent choice for beginners and experienced programmers alike. It supports multiple programming paradigms, including procedural, object-oriented, and functional programming.Python has an extensive standard library that provides pre-built modules and functions for various tasks, reducing the need for developers to write code from scratch. This library covers everything from web development and data analysis to scientific computing and artificial intelligence.
Python's cross-platform compatibility means it can run on various operating systems, fostering a broad community of developers. It's widely used in web development with frameworks like Django and Flask, in scientific computing with libraries such as NumPy and SciPy, and in data analysis and visualization through tools like Pandas and Matplotlib.
Additionally, Python's versatility extends to scripting, automation, and even emerging fields like machine learning and data science, where libraries like TensorFlow and scikit-learn are highly popular. Its opensource nature and active community contribute to its continual evolution and the development of a vast ecosystem of third-party packages and resources. Overall, Python's simplicity, power, and flexibility make it an excellent choice for a diverse array of programming needs.
 
5.3.2	Advantages of Python


•	Readability: Python's clean and easily readable syntax, with its use of indentation, makes it an excellent choice for both beginners and experienced programmers.
•	Wide Community Support: Python boasts a large and active community, which means you can find extensive libraries, frameworks, and resources, making development more efficient.
•	Cross-Platform Compatibility: Python is available on various platforms, ensuring that your code can run on different operating systems without significant modifications.
•	Extensive Standard Library: Python's extensive standard library covers a wide range of modules for common tasks, reducing the need to write code from scratch.
•	High Productivity: Python's simplicity and conciseness lead to faster development cycles and quicker prototyping, saving time and effort.
•	Great for Data Science: Python is a go-to language for data science and machine learning due to libraries like NumPy, Pandas, and TensorFlow.
•	Web Development: Python is used for web development, with frameworks like Django and Flask, making it easy to create web applications.
 
5.4	Gradio

5.4.1	About Gradio
Gradio is a Python library that makes it easy to create simple web interfaces for machine learning models. It helps users interact with models without needing complex coding or web development skills. With Gradio, you can quickly test and share your model using a web-based UI.

5.4.2	Working of Gradio
In a credit card fraud detection system, Gradio can be used to build an easy-to-use interface where users can enter transaction details and get real-time fraud predictions.
	Simple Interface: Users can input transaction details like amount, time, and location to check if a transaction is fraud or not.
	Quick Model Deployment: Instead of setting up a complex website, Gradio provides a fast way to run the fraud detection model and share it via a web link.
	Real-Time Testing: Users can test the model with new transactions and get immediate results.
	Easy Visualization: Gradio can display prediction confidence and explain why a transaction is marked as fraud.
	Fast Prototyping: Developers can quickly test and improve the fraud detection model without extra coding.

5.4.3	Advantages of Gradio

•	Easy to Use & Quick Deployment : With just a few lines of code, Gradio allows you to create an interactive fraud detection interface without needing a complex setup.

•	No Need for a Separate Frontend: Gradio provides a built-in web-based UI where users can input transaction details and get fraud predictions instantly, saving development time.

•	Real-Time Model Testing: You can test your XGBoost or LSTM fraud detection model with live transaction data and immediately see results, helping in model evaluation.

•	Easily Shareable: Gradio generates a public link that allows banks or analysts to test the fraud detection system without setting up a full server.

•	Secure & Local Hosting: The model can be run locally, ensuring that sensitive transaction data remains private and is not exposed to external servers.
 
5.5	SMOTE (Synthetic Minority Over-sampling Technique)



SMOTE (Synthetic Minority Over-sampling Technique) is a method used to handle imbalanced datasets by generating synthetic data points for the minority class. In fraud detection, fraudulent transactions are much fewer than legitimate ones, making it difficult for machine learning models to learn fraud patterns properly. SMOTE helps balance the dataset by creating artificial fraud cases, ensuring the model gets enough examples to learn from.

5.5.1	Advantages of SMOTE

•	Generates Synthetic Samples: Creates new, realistic data points for the minority class instead of duplicating existing ones.
•	Improves Model Performance: Enhances recall, precision, and F1-score, especially for imbalanced datasets.
•	Reduces Overfitting:Avoids repeating the same data, which helps models generalize better.
•	Balances the Dataset:Helps machine learning algorithms treat both classes equally during training.
•	Flexible	Sampling	Strategy:You	can	control	how	much	oversampling	is	done	using sampling_strategy.
•	Works with Most ML Models: Compatible with XGBoost, Random Forest, SVM, and deep learning models like LSTM.
•	Available in Libraries:Easily accessible via tools like imblearn (SMOTE in imbalanced-learn).
 

6.	 IMPLEMENTATION

Implementation Steps for Fraud Detection System

6.1	Load and Explore the Dataset:


•	Import necessary libraries (pandas, matplotlib.pyplot, seaborn, etc.).
•	Load the dataset using pd.read_csv("file_path.csv").
•	Perform exploratory data analysis (EDA) using:.head(), df.tail(), df.describe(), df.info(), df.shape(), df.nunique(), etc.
•	Check for missing values: df.isnull().sum().
•	Check for duplicate records: df.duplicated().sum().


6.2	Fraud Detection Based on Different Features:

	Based on Credit Score:
•	Classify transactions as fraudulent (Credit Score < 500) or non-fraudulent (Credit Score >= 500).
•	Create a new column df["Fraudulent"].
•	Visualize the distribution using a pie chart (matplotlib).
	Based on User Age:
•	Classify fraud transactions where User Age < 18.
•	Generate a pie chart to represent fraud vs. non-fraud transactions.
	Based on IP Address:
•	Generate synthetic IP addresses and assign random fraud labels.
•	Plot the fraud distribution using a pie chart.
	Based on Transaction Amount:
•	Classify fraud transactions where Transaction Amount > 30,000.
•	Use pie and bar charts (seaborn) to visualize fraud distribution
 
6.3	Machine Learning Model: LSTM + XGBoost:

6.3.1	Data Preprocessing:
•	Drop unnecessary columns like Transaction ID, User ID, Timestamp, and IP Address.
•	Encode categorical variables using LabelEncoder().
•	Normalize numerical features using MinMaxScaler().
•	Handle class imbalance using SMOTE.
6.3.2	LSTM Model for Feature Extraction:
•	Define an LSTM model with Sequential() and tensorflow.keras.layers:
•	LSTM layers for sequential learning.
•	Batch normalization and dropout layers to prevent overfitting.
•	Dense output layer with sigmoid activation for binary classification.
•	Train the model with model.fit() and extract LSTM features.
6.3.3	XGBoost for Final Prediction:
•	Train an XGBoost model using the extracted LSTM features.
•	Evaluate model performance using:



6.3.4	Save Trained Models:
•	Save the LSTM model using lstm_model.save("lstm_model.h5").
•	Save the XGBoost model using joblib.dump(xgb_model, "xgb_model.pkl"

6.4	Deploying Fraud Detection with Gradio:


6.4.1	Create a Gradio Interface:
Install gradio (!pip install gradio).
6.4.2	Define a function predict() that:
•	Accepts transaction details as input.
•	Checks for fraud conditions (amount > 30,000, credit score < 500).
•	Returns a fraud prediction with probability.
6.4.3	Create a Gradio interface with:
•	Textbox for Transaction ID.
•	Number inputs for Amount and Credit Score.
•	Outputs showing Prediction and Probability.
 
•	Launch the Gradio app using iface.launch().


 
7.	 TESTING


7.4	Purpose of Testing

The purpose of testing is to discover errors. Testing is the process of trying to discover every conceivable fault or weakness in a work product. It provides a way to check the functionality of components, subassemblies, assemblies and/or a finished product. It is the process of exercising software with the intent of ensuring that the Software system meets its requirements and user expectations and does not fail in an unacceptable manner. There are various types of tests. Each test type addresses a specific testing requirement.


7.5	Steps for Testing Process:


7.5.1	Data Validation Testing:
•	Ensure the dataset is properly loaded.
•	Check for missing values (df.isnull().sum()).
•	Verify that categorical data is correctly encoded.
•	Check for duplicate records (df.duplicated().sum()).


7.5.2	Model Training & Evaluation Testing:
•	Verify that the dataset is split correctly into training and testing sets.
•	Ensure SMOTE is applied properly to balance the dataset.
•	Validate LSTM model training using loss/accuracy metrics.
•	Evaluate XGBoost classifier with performance metrics:
•	Accuracy Score
•	Precision, Recall, F1-score
•	Confusion Matrix
 
7.5 3. Feature Engineering Testing:
•	Check if categorical variables (Payment Method, Device Type, Merchant Category) are correctly label-encoded.
•	Ensure numerical features are scaled correctly using MinMaxScaler.
•	Verify the transformation of input data into LSTM-compatible format.


7.5.4 Prediction Testing:
•	Test real-time predictions with different input values.
•	Ensure the model correctly predicts fraud based on:
•	Credit Score (< 500 → Fraud)
•	Transaction Amount (> 30,000 → Fraud)
•	User Age (< 18 → Fraud)
•	High-Risk Country flag


7.5.5 Gradio Interface Testing:
•	Test input variations for fraud and non-fraud cases.
•	Validate probability output correctness.


7.5.6. Edge Case Testing:
•	Input boundary values for Credit Score (300, 850), Age (18, 100), and Transaction Amount (0, 50,000).
•	Test with missing or incorrect values.
•	Verify system handles invalid data gracefully.


7.6	Performance Testing:
•	Measure prediction response time.
•	Check for memory usage and model load times.


 
9.	CONCLUSION

This project successfully implemented a credit card fraud detection system using LSTM (Long Short-Term Memory) and XGBoost models. By leveraging real-time transaction data, the system identifies fraudulent transactions based on multiple factors such as credit score, transaction amount, user age, and geographical risk. The combination of LSTM for feature extraction and XGBoost for classification significantly improved prediction accuracy. Additionally, SMOTE (Synthetic Minority Over-sampling Technique) was used to address class imbalance, ensuring the model remains effective even with skewed datasets.
The results demonstrated high accuracy and balanced classification performance, showing the system's ability to detect fraudulent transactions while minimizing false positives. The Streamlit-based web application and Gradio interface further enhanced user accessibility, allowing real-time predictions for financial institutions and customers.


 
10.	 FUTURE WORK

In the future, this fraud detection system can be improved by using LSTMs or Transformers to better detect fraud in transaction patterns. Additionally, unsupervised learning methods, such as anomaly detection, can help identify new fraud types without needing labeled data. Another useful approach is Graph Neural Networks (GNNs), which can detect fraud by analyzing connections between suspicious users. These improvements will make fraud detection more accurate and adaptable to new fraud tactics.

Another important step is making the system faster and easier to use in real-world applications. Deploying it as a cloud-based API will allow banks and businesses to check transactions for fraud in real time. Using blockchain technology can improve security by keeping tamper-proof records of transactions. Additionally, making the AI more explainable will help users and regulators understand why a transaction is marked as fraud. Lastly, adding user feedback and self-learning models will help the system improve over time, making fraud detection smarter and more reliable.


 
11.	REFERENCES

[1]	Credit,Card, , Fraud, , Detection, , Based, , on, , Transaction, , Behavior, , -by, John, Richard, Larry, A., Vea”, published, by, Proc., of, , the, 2017, , IEEE, Region, , 10, , Conference, (TENCON)
Malaysia, November, , 5-8,, 2017
[2]	CLIFTON, PHUA1,, VINCENT, LEE1,, KATE, SMITH1, &, ROSS, GAYLER2, “, , A,
Comprehensive, , Survey, of, , Data, Mining-based, , Fraud, Detection, Research”, published, by, School, of, Business, Systems,, Faculty, of, Information, , Technology,, Monash, , University, Wellington, , Road,, Clayton,, Victoria, 3800,, Australia,
[3]	“Survey, Paper, on, Credit, Card, Fraud, Detection, by, Suman”, Research, Scholar, ,GJUS&T, Hisar, , HCE,, Sonepat, , published, by, , International, Journal, of, Advanced, Research, in, Computer, Engineering, &, Technology, (IJARCET), Volume, 3, Issue, 3,, March, 2014.
[4]	Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence Research, 16, 321-357. https://doi.org/10.1613/jair.953
[5]	Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16), 785-794. https://doi.org/10.1145/2939672.2939785
[6]	Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735-1780. https://doi.org/10.1162/neco.1997.9.8.1735
[7]	Dal Pozzolo, A., Boracchi, G., Caelen, O., Alippi, C., & Bontempi, G. (2018). Credit Card Fraud Detection: A Realistic Modeling and a Novel Learning Strategy. IEEE Transactions on Neural Networks and Learning Systems, 29(8), 3784-3797. https://doi.org/10.1109/TNNLS.2017.2736643
[8]	Carcillo, F., Dal Pozzolo, A., Le Borgne, Y.-A., Caelen, O., Mazzer, Y., & Bontempi, G. (2019). Combining Unsupervised and Supervised Learning in Credit Card Fraud Detection. Information Sciences, 479, 448-460. https://doi.org/10.1016/j.ins.2018.05.042
 
