#  IBM Applied Data Science Capstone: SpaceX Launch Prediction Project

### Predicting SpaceX First Stage Landings with Data Science  
*A real-world capstone project integrating data collection, wrangling, visualization, and machine learning.*


##  Project Background & Motivation

**SpaceX** has transformed space technology by introducing reusable rockets, significantly reducing launch costs. For example, the **Falcon 9** rocket, priced at ~$62 million per launch, is considerably cheaper than competitors due to its reusable first stage.

Accurately predicting landing success can optimize mission planning, reduce risk, and help manage costs. This project aims to explore **whether the Falcon 9 first stage will land successfully** using historical launch data.

---

##  Research Objectives

- How do variables like **payload mass**, **launch site**, **orbit type**, and **flight number** influence landing success?
- Has the **rate of successful landings** improved over time?
- Which **machine learning algorithm** best predicts binary landing outcomes (success/failure)?

---

## ⚙ Methodology

###  Data Collection
- Collected SpaceX launch data using the **SpaceX REST API**
- Scraped additional information from **Wikipedia**

###  Data Wrangling
- Filtered, cleaned, and merged datasets
- Handled missing values and duplicates
- Applied **One-Hot Encoding** for categorical variables

###  Exploratory Data Analysis (EDA)
- Visualized trends using **Matplotlib**, **Seaborn**, and **SQL**
- Performed in-depth exploration of launch sites, payload distribution, and success rates

###  Interactive Analytics
- Developed **Folium** maps to show global launch site distribution and proximity analysis
- Built a **Plotly Dash dashboard** with interactive components to explore data dynamically

###  Predictive Modeling
- Trained and evaluated multiple classification models:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Decision Tree
  - K-Nearest Neighbors (KNN)
- Performed **GridSearchCV** for hyperparameter tuning
- Selected the best-performing model based on **accuracy and confusion matrix**

---

##  Repository Structure

- `notebooks/`: Jupyter Notebooks with step-by-step analysis and modeling
- `dash_app/`: Source code for the Plotly Dash interactive dashboard
- `visuals/`: Screenshots and charts used in the presentation
- `data/`: Cleaned and processed datasets
- `README.md`: Overview and documentation of the project

---

##  Use Case

This repository serves as a **learning tool** for students and professionals pursuing the IBM Data Science Capstone. It provides a reference for tackling real-world data science challenges with a structured workflow.

**Note:** This project is intended for educational purposes only. Please do not use this repository to bypass coursework or violate academic integrity.

---

##  License

This project is licensed under the [GNU GPL v3.0](https://www.gnu.org/licenses/gpl-3.0.html).  
Feel free to use, modify, and share the contents for **non-commercial** purposes with proper attribution.

---

##  Contributing

Contributions are welcome!  
If you have suggestions, improvements, or additional insights relevant to this capstone, feel free to submit a pull request.
