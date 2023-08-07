# Pima-Indian-Women-Diabets-Feature-Engineering
1. **Importing Libraries and Data**: Importing the necessary libraries and reading a CSV file containing diabetes-related data.

2. **Data Exploration and Analysis**:
   - Checking the basic information about the dataset using functions like `check_df`.
   - Identifying and categorizing columns as numerical, categorical, etc. using the `grab_col_names` function.
   - Summarizing numerical columns using the `num_summary` function.
   - Analyzing the relationship between numerical columns and the target variable using the `target_summary_with_num` function.
   - Creating a correlation matrix and visualizing it using a heatmap.

3. **Data Preprocessing and Cleaning**:
   - Splitting the dataset into features (X) and target variable (y).
   - Splitting the data into training and testing sets using `train_test_split`.
   - Fitting a RandomForestClassifier model to the training data and making predictions on the test data.
   - Evaluating the model's performance using metrics like accuracy, recall, precision, F1-score, and ROC AUC.

4. **Handling Missing Values and Outliers**:
   - Handling missing values by replacing zeros and filling with the mean.
   - Identifying and handling outliers using the `outlier_thresholds`, `check_outlier`, and `replace_with_tresholds` functions.

5. **Feature Engineering**:
   - Creating new features based on age, blood pressure, BMI, glucose level, and other criteria.
   - Encoding categorical variables using label encoding and one-hot encoding.
   - Applying standard scaling to numerical features.

6. **Model Training and Evaluation**:
   - Splitting the preprocessed data into training and testing sets.
   - Training a RandomForestClassifier model on the training data.
   - Making predictions on the test data and evaluating model performance using various metrics.
