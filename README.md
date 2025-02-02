# Predictive_analysis
COMPANY : CODTECH IT SOLUTIONS 
NAME : JASVANTH VARMA MUPPALA 
INTERN ID : CT08ORY 
DOMAIN : DATA ANALYSIS 
DURATION : 4 WEEKS 
MENTOR : NEELA SANTOSH


### Project Description: **NYC Taxi Zones Classification**

#### **Overview:**
This project aims to predict the `Borough` of a New York City taxi zone based on the zone's characteristics, such as its `LocationID`, `Zone`, and `service_zone`. Using machine learning, specifically a Random Forest classifier, the project analyzes data from the `nyc_taxi_zones.csv` dataset, which contains information about various taxi zones in New York City.

#### **Key Steps:**
1. **Dataset Loading:**
   - The project begins by loading a dataset (`nyc_taxi_zones.csv`) using the `pandas` library. This dataset contains details such as `LocationID`, `Borough`, `Zone`, and `service_zone`.

2. **Data Preprocessing:**
   - The dataset is preprocessed to handle missing values in important columns (`Borough`, `Zone`, `service_zone`). Missing values are filled with placeholders (`'Unknown'`) to ensure there are no gaps that could lead to errors in the model.
   - Categorical variables (`Zone` and `service_zone`) are transformed using `LabelEncoder`, converting them into numerical values so that they can be used as input for machine learning algorithms.

3. **Feature Selection:**
   - The features (`LocationID`, `Zone`, `service_zone`) are selected to train the model. These features provide information about each zone that can be used to predict the corresponding `Borough`.

4. **Target Variable Encoding:**
   - The target variable, `Borough`, which represents the geographical borough of the taxi zone (e.g., Bronx, Manhattan, etc.), is encoded into numerical values using `LabelEncoder`.

5. **Model Training:**
   - The dataset is split into training and test sets using `train_test_split` from scikit-learn. A Random Forest classifier is trained on the training data (`X_train`, `y_train`), which involves constructing an ensemble of decision trees to learn the relationship between the features and the target variable.

6. **Model Evaluation:**
   - After the model is trained, predictions are made on the test set (`X_test`), and the accuracy is evaluated using `accuracy_score`. The classification performance is also evaluated using `classification_report`, which provides precision, recall, and F1 scores for each class.

7. **Model Saving:**
   - Finally, the trained model and label encoders are saved using `pickle`, allowing the model to be reused in the future for making predictions without retraining.

#### **Technologies Used:**
- **Python**: The primary programming language for data manipulation, machine learning model training, and evaluation.
- **VS Code**: Used as the integrated development environment (IDE) for writing and running the Python code. VS Code provides useful extensions like Python and Jupyter for an efficient coding experience.
- **Libraries**: 
  - `pandas`: For loading and manipulating the dataset.
  - `numpy`: For numerical operations.
  - `scikit-learn`: For machine learning tasks such as data preprocessing, model training, and evaluation.
  - `pickle`: For saving the trained model and encoders.

#### **Project Outcome:**
The project classifies taxi zones into their respective boroughs based on available features, achieving an accuracy of approximately 41.51%. Although the model could be further fine-tuned for better performance, it demonstrates the ability to predict categorical data based on spatial features. The model and encoders are saved for future use in deployment or further analysis.

#### **Challenges Encountered:**
- **Missing Data**: The dataset had missing values, which were handled by filling them with a placeholder. This approach may affect the model's performance, and alternative imputation strategies could be explored.
- **Model Performance**: With an accuracy of around 41.51%, the Random Forest model could benefit from hyperparameter tuning, more features, or additional data to improve its prediction accuracy.


# OUTPUT :

Dataset Sample:
    LocationID        Borough                     Zone service_zone
0           1            EWR           Newark Airport          EWR
1           2         Queens              Jamaica Bay    Boro Zone
2           3          Bronx  Allerton/Pelham Gardens    Boro Zone
3           4      Manhattan            Alphabet City  Yellow Zone
4           5  Staten Island            Arden Heights    Boro Zone

Missing values in the dataset:
 LocationID      0
Borough         1
Zone            1
service_zone    2
dtype: int64

Accuracy: 0.4151

Classification Report:
               precision    recall  f1-score   support

      Bronx       0.45      0.53      0.49       102
     Brooklyn     0.40      0.45      0.42        89
  Manhattan      0.35      0.34      0.35        91
      Queens      0.52      0.45      0.48       104
 Staten Island   0.37      0.45      0.41       100

    accuracy                           0.42       486
   macro avg       0.42      0.44      0.42       486
weighted avg       0.42      0.42      0.42       486
