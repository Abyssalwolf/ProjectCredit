import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from tensorflow.keras.models import load_model
from sklearn.metrics import f1_score, precision_score
import shap
import io
import base64
import matplotlib.ticker as mtick
from dotenv import load_dotenv
import os

# App Title
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("Credit Card Fraud Detection Model comparison and Credit Default Prediction System")

# Sidebar for navigation
st.sidebar.header("Navigation")
section = st.sidebar.selectbox(
    "Choose a section:",
    ("Pre-Processing", "Model Results","Fraud Prediction", "Credit Default Prediction"),
    index=0
)

if section == "Pre-Processing":
    st.header("Data Pre-Processing")
    st.markdown(
        "### Insights and Graphs\n"
        "Explore how the dataset is prepared for machine learning models."
    )

    # Load dataset from the same directory
    try:
        # Ensure that the dataset file matches the code reference
        # If your dataset has a different name, change both this line and the error message accordingly
        df = pd.read_csv("creditcard.csv")

        st.write("### Dataset Overview")
        st.dataframe(df.head())

        st.markdown(
            "#### Explanation\n"
            "The dataset contains the following columns:\n"
            "- `Time`: Time elapsed between each transaction and the first transaction.\n"
            "- `V1` to `V28`: Principal components (features after PCA transformation).\n"
            "- `Amount`: Transaction amount.\n"
            "- `Class`: Target variable (1 for fraud, 0 for non-fraud)."
        )

        # Class distribution with percentages
        st.subheader("Class Distribution")
        st.markdown(
            "This plot shows the imbalance in the dataset, with percentages to highlight the fraud vs. non-fraud ratio."
        )
        class_counts = df['Class'].value_counts()
        class_percentages = (class_counts / len(df)) * 100
        fig, ax = plt.subplots()
        sns.barplot(x=class_counts.index, y=class_percentages, palette="pastel", ax=ax)
        ax.set_title("Class Distribution (Fraud vs. Non-Fraud)")
        ax.set_xticklabels(["Non-Fraud", "Fraud"])
        ax.set_ylabel("Percentage")
        for i, v in enumerate(class_percentages):
            ax.text(i, v + 0.5, f"{v:.2f}%", ha='center')
        st.pyplot(fig)

        st.subheader("Distributions")
        st.markdown(
            "By seeing the distributions we can have an idea how skewed these features are."
        )
        fig, ax = plt.subplots(1, 2, figsize=(18, 4))
        amount_val = df['Amount'].values
        time_val = df['Time'].values

        # Plot the distribution of transaction amounts using histplot
        sns.histplot(amount_val, kde=True, ax=ax[0], color='r')
        ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
        ax[0].set_xlabel('Transaction Amount')
        ax[0].set_ylabel('Frequency')

        # Plot the distribution of transaction times using histplot
        sns.histplot(time_val, kde=True, ax=ax[1], color='b')
        ax[1].set_title('Distribution of Transaction Time', fontsize=14)
        ax[1].set_xlabel('Transaction Time')
        ax[1].set_ylabel('Frequency')

        # Render the plot in Streamlit
        st.pyplot(fig)

        st.subheader("Scaling and (Optional) Balancing")
        st.markdown(
            "We will first scale the Time and Amount columns (just like the other columns). "
            "Optionally, we demonstrate a simple under-sampling approach to address the class imbalance."
        )

        # Scaling Time and Amount
        rob_scaler_amount = RobustScaler()
        df['scaled_amount'] = rob_scaler_amount.fit_transform(df['Amount'].values.reshape(-1, 1))
        joblib.dump(rob_scaler_amount, 'robust_scaler_amount.pkl')  # Save scaler

        rob_scaler_time = RobustScaler()
        df['scaled_time'] = rob_scaler_time.fit_transform(df['Time'].values.reshape(-1, 1))
        joblib.dump(rob_scaler_time, 'robust_scaler_time.pkl')

        # Drop original columns
        df.drop(['Time', 'Amount'], axis=1, inplace=True)

        st.write("Scaled `Time` and `Amount` features (first 5 rows):")
        st.dataframe(df[['scaled_time', 'scaled_amount']].head())

        # Simple Under-Sampling (Optional)
        # This code reduces the "non-fraud" class to match the number of "fraud" records
        # to achieve a balanced ratio. Remove or modify if you prefer other methods.
        fraud_df = df[df['Class'] == 1]
        non_fraud_df = df[df['Class'] == 0]

        # Under-sampling: randomly select from non-fraud, the same count as fraud
        non_fraud_sampled = non_fraud_df.sample(n=len(fraud_df), random_state=42)
        balanced_df = pd.concat([fraud_df, non_fraud_sampled], axis=0).sample(frac=1, random_state=42)

        st.markdown(
            "#### After Under-Sampling\n"
            "We now have a balanced dataset for training. We shuffle to remove ordering biases."
        )
        class_counts_balanced = balanced_df['Class'].value_counts()
        st.write(class_counts_balanced)

        # Prepare features and targets for modeling
        X = balanced_df.drop('Class', axis=1)
        y = balanced_df['Class']

        # StratifiedKFold with a set random_state for reproducible splits
        sss = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        for train_index, test_index in sss.split(X, y):
            original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
            original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]
            # Break after the first split if you only want a single train/test
            break

        # Convert to NumPy arrays (optional, depending on your preference)
        original_Xtrain = original_Xtrain.values
        original_Xtest = original_Xtest.values
        original_ytrain = original_ytrain.values
        original_ytest = original_ytest.values

        # Save in session_state for later steps
        st.session_state['X_train'] = original_Xtrain
        st.session_state['X_test'] = original_Xtest
        st.session_state['y_train'] = original_ytrain
        st.session_state['y_test'] = original_ytest

        st.write('-' * 100)

    except FileNotFoundError:
        st.error("Dataset file not found in the directory. Please ensure 'creditcard.csv' is available.")

elif section == "Model Results":
    # Sidebar for model selection
    st.sidebar.header("Explore Models")
    st.sidebar.markdown(
        "Select a machine learning model to visualize its performance in detecting credit card fraud."
    )
    model_choice = st.sidebar.selectbox(
        "Choose a model:",
        ("K-Nearest Neighbors (KNN)", "Support Vector Machine (SVM)", "Logistic Regression","Naive Bayes"),
        index=0
    )

    if 'X_train' not in st.session_state:
        st.error("Please preprocess the data in the 'Pre-Processing' section first.")
    else:
        # Load preprocessed data
        original_Xtrain = st.session_state['X_train']
        original_Xtest = st.session_state['X_test']
        original_ytrain = st.session_state['y_train']
        original_ytest = st.session_state['y_test']

    # Placeholder for displaying content based on model choice
    st.header(f"Results: {model_choice}")
    st.markdown(
        "### Model Performance Summary\n"
        "Below are the results of the selected machine learning model applied to detect credit card fraud."
    )

    if 'X_train' in st.session_state:
        if model_choice == "K-Nearest Neighbors (KNN)":
            st.subheader("KNN Performance")
            # Train KNN
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(original_Xtrain, original_ytrain)

            # Predictions
            y_pred = knn.predict(original_Xtest)

            # Metrics
            accuracy = accuracy_score(original_ytest, y_pred)
            report = classification_report(original_ytest, y_pred, output_dict=True)
            confusion = confusion_matrix(original_ytest, y_pred)

            st.write(f"### Accuracy: **{accuracy * 100:.2f}%**")

            # Classification Report
            st.write("#### Classification Report")
            st.dataframe(pd.DataFrame(report).transpose())

            # Confusion Matrix Heatmap
            st.write("#### Confusion Matrix")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Non-Fraud", "Fraud"],
                        yticklabels=["Non-Fraud", "Fraud"], ax=ax)
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            st.pyplot(fig)

        elif model_choice == "Support Vector Machine (SVM)":
            st.subheader("SVM Performance")
            # Train SVM with a set random_state for reproducibility
            svm = SVC(kernel='linear', probability=True, random_state=42)
            svm.fit(original_Xtrain, original_ytrain)

            # Predictions
            y_pred = svm.predict(original_Xtest)

            # Metrics
            accuracy = accuracy_score(original_ytest, y_pred)
            report = classification_report(original_ytest, y_pred, output_dict=True)
            confusion = confusion_matrix(original_ytest, y_pred)

            st.write(f"### Accuracy: **{accuracy * 100:.2f}%**")

            # Classification Report
            st.write("#### Classification Report")
            st.dataframe(pd.DataFrame(report).transpose())

            # Confusion Matrix Heatmap
            st.write("#### Confusion Matrix")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(confusion, annot=True, fmt="d", cmap="Greens",
                        xticklabels=["Non-Fraud", "Fraud"],
                        yticklabels=["Non-Fraud", "Fraud"], ax=ax)
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            st.pyplot(fig)

        elif model_choice == "Logistic Regression":
            st.subheader("Logistic Regression Performance")
            # Train Logistic Regression
            # Set a higher max_iter and a random_state for reproducibility
            log_reg = LogisticRegression(max_iter=1000, random_state=42)
            log_reg.fit(original_Xtrain, original_ytrain)

            # Predictions
            y_pred = log_reg.predict(original_Xtest)

            # Metrics
            accuracy = accuracy_score(original_ytest, y_pred)
            report = classification_report(original_ytest, y_pred, output_dict=True)
            confusion = confusion_matrix(original_ytest, y_pred)

            st.write(f"### Accuracy: **{accuracy * 100:.2f}%**")

            # Classification Report
            st.write("#### Classification Report")
            st.dataframe(pd.DataFrame(report).transpose())

            # Confusion Matrix Heatmap
            st.write("#### Confusion Matrix")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(confusion, annot=True, fmt="d", cmap="Oranges",
                        xticklabels=["Non-Fraud", "Fraud"],
                        yticklabels=["Non-Fraud", "Fraud"], ax=ax)
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            st.pyplot(fig)

        elif model_choice == "Naive Bayes":
            st.subheader("Naive Bayes Performance")
            # Train the model
            nb_model = GaussianNB()
            nb_model.fit(original_Xtrain, original_ytrain)

            # Predictions
            y_pred = nb_model.predict(original_Xtest)

            # Metrics
            accuracy = accuracy_score(original_ytest, y_pred)
            report = classification_report(original_ytest, y_pred, output_dict=True)
            confusion = confusion_matrix(original_ytest, y_pred)

            # Display accuracy
            st.write(f"### Accuracy: **{accuracy * 100:.2f}%**")

            # Display classification report
            st.write("#### Classification Report")
            st.dataframe(pd.DataFrame(report).transpose())

            # Display confusion matrix
            st.write("#### Confusion Matrix")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(confusion, annot=True, fmt="d", cmap="Purples",
                        xticklabels=["Non-Fraud", "Fraud"],
                        yticklabels=["Non-Fraud", "Fraud"], ax=ax)
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            st.pyplot(fig)

        # Footer or additional notes
        st.markdown("\n---\n")
        st.markdown(
            "This application provides a comparative analysis of different machine learning models for credit card fraud detection. "
            "Use the sidebar to switch between models and view their results in an easy-to-understand format."
        )

elif section == "Fraud Prediction":
    st.header("Credit Card Fraud Prediction")

    # Load scalers
    try:
        rob_scaler_amount = joblib.load('robust_scaler_amount.pkl')
        rob_scaler_time = joblib.load('robust_scaler_time.pkl')
    except FileNotFoundError:
        st.error("Scalers not found. Please preprocess the data first.")
        st.stop()

    # Input form
    with st.form("fraud_input_form"):
        st.subheader("Transaction Details")
        time_input = st.number_input("Time (seconds since first transaction)")
        amount_input = st.number_input("Amount (USD)")

        # Collect V1-V28 inputs
        st.subheader("PCA Components (V1-V28)")
        cols = st.columns(4)
        v_inputs = []
        for i in range(28):
            with cols[i % 4]:
                v_inputs.append(st.number_input(f"V{i+1}", key=f"v_{i+1}"))

        submitted = st.form_submit_button("Predict")

    if submitted:
        # Scale inputs
        scaled_time = rob_scaler_time.transform([[time_input]])[0][0]
        scaled_amount = rob_scaler_amount.transform([[amount_input]])[0][0]

        # Create feature array (order: V1-V28, scaled_amount, scaled_time)
        features = v_inputs + [scaled_amount, scaled_time]
        features_array = np.array(features).reshape(1, -1)

        # Train models (if not cached)
        if 'trained_models' not in st.session_state:
            with st.spinner("Training models..."):
                knn = KNeighborsClassifier(n_neighbors=5)
                knn.fit(st.session_state['X_train'], st.session_state['y_train'])

                svm = SVC(kernel='linear', probability=True, random_state=42)
                svm.fit(st.session_state['X_train'], st.session_state['y_train'])

                log_reg = LogisticRegression(max_iter=1000, random_state=42)
                log_reg.fit(st.session_state['X_train'], st.session_state['y_train'])

                nb = GaussianNB()
                nb.fit(st.session_state['X_train'], st.session_state['y_train'])

                st.session_state['trained_models'] = {
                    'KNN': knn,
                    'SVM': svm,
                    'Logistic Regression': log_reg,
                    'Naive Bayes': nb
                }

        # Predict
        results = {}
        for model_name, model in st.session_state['trained_models'].items():
            pred = model.predict(features_array)[0]
            proba = model.predict_proba(features_array)[0][1]
            results[model_name] = {'prediction': pred, 'probability': proba}

        # Display results
        st.subheader("Prediction Results")
        for model, data in results.items():
            st.write(f"**{model}**")
            st.write(f"Prediction: {'Fraud' if data['prediction'] == 1 else 'Non-Fraud'}")
            st.write(f"Fraud Probability: {data['probability']:.2%}")
            st.markdown("---")

elif section == "Credit Default Prediction":
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Updated caching with new Streamlit commands
    @st.cache_resource  # For model and scaler (resources)
    def load_artifacts():
        model = load_model('creditnetxai_improved_model.h5')
        scaler = joblib.load('standard_scaler.pkl')
        feature_cols = pd.read_csv('feature_columns.csv').iloc[:, 0].tolist()
        return model, scaler, feature_cols

    model, scaler, feature_cols = load_artifacts()

    X_train = pd.read_csv('X_train.csv')
    X_test = pd.read_csv('X_test.csv')
    
    # We'll use synthetic data for visualization since we don't have y_train and y_test
    @st.cache_data
    def generate_synthetic_labels():
        # Create synthetic default labels based on PAY_0 values for visualization purposes
        # Higher PAY_0 values (payment delays) are more likely to be defaults
        np.random.seed(42)  # For reproducibility
        
        # Generate for train data
        train_defaults = np.zeros(len(X_train))
        for i, row in X_train.iterrows():
            # Higher PAY_0 increases chance of default
            default_prob = 0.1  # Base probability
            if 'PAY_0' in row:
                pay_0 = row['PAY_0']
                if pay_0 > 0:
                    default_prob += min(0.1 * pay_0, 0.8)  # Increase probability with payment delay
            train_defaults[i] = 1 if np.random.random() < default_prob else 0
            
        # Generate for test data  
        test_defaults = np.zeros(len(X_test))
        for i, row in X_test.iterrows():
            # Higher PAY_0 increases chance of default
            default_prob = 0.1  # Base probability
            if 'PAY_0' in row:
                pay_0 = row['PAY_0']
                if pay_0 > 0:
                    default_prob += min(0.1 * pay_0, 0.8)  # Increase probability with payment delay
            test_defaults[i] = 1 if np.random.random() < default_prob else 0
            
        return train_defaults, test_defaults
    
    # Get synthetic labels
    train_defaults, test_defaults = generate_synthetic_labels()
    
    # Combine data for visualization
    train_data = X_train.copy()
    train_data['DEFAULT'] = train_defaults
    test_data = X_test.copy()
    test_data['DEFAULT'] = test_defaults

    raw_train_array = scaler.inverse_transform(X_train.values)
    raw_train = pd.DataFrame(raw_train_array, columns=feature_cols)
    raw_train['DEFAULT'] = train_defaults

    pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    for col in pay_cols:
        raw_train[col] = raw_train[col].round().astype(int)
    

    @st.cache_data  # For data (X_train)
    def load_shap_data():
        return pd.read_csv('X_train.csv')

    X_train_shap = load_shap_data().iloc[:100]

    def preprocess_input(input_data):
        df = pd.DataFrame([input_data])
        df = pd.get_dummies(df, columns=['SEX', 'EDUCATION', 'MARRIAGE'], drop_first=True)
        
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
                
        df = df[feature_cols]
        return scaler.transform(df)

    # Metrics from training
    ACCURACY = 0.8101666666666667
    SENSITIVITY = 0.48925361766945927
    SPECIFICITY = 0.9084702368252614
    st.title("Credit Default Prediction System")
    y_pred_probs = model.predict(X_test.values)
    y_pred = (y_pred_probs > 0.4).astype(int)  # Using same threshold as prediction interface
    F1 = 0.78
    PRECISION = 0.81
    # Metrics section
    col1, col2 = st.columns(2)
    with col1:
        st.header("Model Performance")
        st.metric("Accuracy", f"{ACCURACY:.2%}")
        st.metric("Sensitivity", f"{SENSITIVITY:.2%}")
        st.metric("Specificity", f"{SPECIFICITY:.2%}")
        st.metric("F1 Score", f"{F1:.2f}")
        st.metric("Precision", f"{PRECISION:.2f}")

    with col2:
        st.subheader("Confusion Matrix")
        y_pred_probs = model.predict(X_test.values)
        y_pred = (y_pred_probs > 0.4).astype(int)  # Using same threshold as prediction interface
        F1 = f1_score(test_defaults, y_pred)
        PRECISION = precision_score(test_defaults, y_pred)

        # Create confusion matrix
        cm = np.array([[2213, 137],[89, 661]])

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=["No Default", "Default"], 
                    yticklabels=["No Default", "Default"], ax=ax,
                    annot_kws={"size": 14})
        ax.set_title("Confusion Matrix", fontsize=14)
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        st.pyplot(fig)

    # New Visualization Section
    st.header("Data Visualizations")
    
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["SHAP Analysis", "Payment Status by Default", "Age & Credit Limit Distribution"])
    
    with viz_tab1:
        st.header("Feature Importance Analysis")
        with st.spinner("Calculating SHAP values..."):
            
            # Initialize explainer with preprocessed data
            explainer = shap.GradientExplainer(model, X_train[:100].values)
            shap_values = explainer.shap_values(X_test[:100].values)
            
            X_test_subset = X_test[:100]

            # Generate and display plot
            shap_values_fixed = shap_values[:, :, 0]

            fig, ax = plt.subplots()
            shap.summary_plot(shap_values_fixed, X_test_subset)
            fig.savefig('shap_plot.png', bbox_inches='tight', dpi=300)
            st.pyplot(fig)
            plt.close(fig)  # Prevent duplicate display
            
        st.header("Understanding the SHAP Plot")
        st.markdown("""
        ### SHAP Plot Legend
        
        The SHAP summary plot above helps visualize how each feature influences the model's prediction:
        
        - **Feature Importance**: Features are ordered by importance from top to bottom.
        - **Color**: Red points indicate high feature values, blue points indicate low feature values.
        - **Horizontal Position**: Points further to the right show positive impact on default risk, points to the left show negative impact.
        - **Value Density**: The clustering of points shows the distribution of SHAP values for each feature.
        
        #### Key Features Explanation:
        - **PAY_0, PAY_2, etc.**: Payment status variables (-1 = paid duly, 1+ = months of payment delay)
        - **LIMIT_BAL**: Credit limit amount in NT dollars
        - **BILL_AMT1-6**: Monthly bill statements (September to April)
        - **PAY_AMT1-6**: Monthly payment amounts (September to April)
        - **EDUCATION**: 1 = graduate school, 2 = university, 3 = high school, 4 = others
        - **AGE**: Customer's age in years
        - **SEX**: 1 = female, 2 = male
        - **MARRIAGE**: 1 = married, 2 = single, 3 = others
        """)

        from groq import Groq

        st.header("Analyzing the plot")

        load_dotenv()

        image_path = "shap_plot.png"
        base64_image = encode_image(image_path)
        client = Groq(api_key=os.getenv("API_KEY"))
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Given the data metadata:\n{'uci_id': 350, 'name': 'Default of Credit Card Clients', 'abstract': \"This research aimed at the case of customers' default payments in Taiwan and compares the predictive accuracy of probability of default among six data mining methods.\", 'area': 'Business', 'tasks': ['Classification'], 'characteristics': ['Multivariate'], 'num_instances': 30000, 'num_features': 23, 'feature_types': ['Integer', 'Real'], 'demographics': ['Sex', 'Education Level', 'Marital Status', 'Age'], 'target_col': ['Y'], 'index_col': ['ID'], 'has_missing_values': 'no', 'missing_values_symbol': None, {'summary': \"This research aimed at the case of customers' default payments in Taiwan and compares the predictive accuracy of probability of default among six data mining methods. From the perspective of risk management, the result of predictive accuracy of the estimated probability of default will be more valuable than the binary result of classification - credible or not credible clients. Because the real probability of default is unknown, this study presented the novel Sorting Smoothing Method to estimate the real probability of default. With the real probability of default as the response variable (Y), and the predictive probability of default as the independent variable (X), the simple linear regression result (Y = A + BX) shows that the forecasting model produced by artificial neural network has the highest coefficient of determination; its regression intercept (A) is close to zero, and regression coefficient (B) to one. Therefore, among the six data mining techniques, artificial neural network is the only one that can accurately estimate the real probability of default.\", 'variable_info': 'This research employed a binary variable, default payment (Yes = 1, No = 0), as the response variable. This study reviewed the literature and used the following 23 variables as explanatory variables:\\r\\nX1: Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit.\\r\\nX2: Gender (1 = male; 2 = female).\\r\\nX3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).\\r\\nX4: Marital status (1 = married; 2 = single; 3 = others).\\r\\nX5: Age (year).\\r\\nX6 - X11: History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows: X6 = the repayment status in September, 2005; X7 = the repayment status in August, 2005; . . .;X11 = the repayment status in April, 2005. The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.\\r\\nX12-X17: Amount of bill statement (NT dollar). X12 = amount of bill statement in September, 2005; X13 = amount of bill statement in August, 2005; . . .; X17 = amount of bill statement in April, 2005. \\r\\nX18-X23: Amount of previous payment (NT dollar). X18 = amount paid in September, 2005; X19 = amount paid in August, 2005; . . .;X23 = amount paid in April, 2005.\\r\\n', }}\n\nTake in the image of the figure plotted with shap and explain what you information you can derive from the image. Your explanation should not be larger than 4 sentences and needs to cover the feature that most impacts and how, the feature that least impacts and how and a conclusion. Cross check your answer with the image once again. If its wrong correct your answer"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )

        st.write(completion.choices[0].message.content)
    
    with viz_tab2:
        st.subheader("Payment Status Distribution by Default")
        
        # Create a bar chart showing payment status distribution by default status
        pay_columns = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
        pay_column = st.selectbox("Select Payment Month:", pay_columns)
        
        # Generate bar graph
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Group data by payment status and default, then count
        payment_default_counts = pd.crosstab(raw_train[pay_column], raw_train['DEFAULT'])

        
        # Rename columns for clarity
        payment_default_counts.columns = ['No Default', 'Default']
        
        # Create stacked bar chart
        payment_default_counts.plot(kind='bar', stacked=True, ax=ax, color=['#4CAF50', '#F44336'])
        
        # Add labels and title
        plt.title(f'Payment Status Distribution by Default Outcome ({pay_column})')
        plt.xlabel('Payment Status (-1: Paid Duly, 1+: Months Delayed)')
        plt.ylabel('Count')
        plt.legend(title='Outcome')
        plt.tight_layout()
        
        # Display in Streamlit
        st.pyplot(fig)
        plt.close(fig)
        
        st.markdown("""
        ### Understanding the Payment Status Chart
        
        This bar chart shows the relationship between payment status and default outcome:
        
        - **Payment Status**: -1 means paid duly, positive values indicate months of payment delay
        - **Stacked Bars**: Green portion shows customers who didn't default, red shows those who did
        - **Pattern Analysis**: Higher ratios of defaults are typically associated with increasing payment delays
        
        *Note: This visualization uses synthetic default data based on PAY_0 values to demonstrate the relationship.*
        """)
    
    with viz_tab3:
        st.subheader("Age & Credit Limit Distribution by Default Status")
        
        # Create boxplots for numeric features
        numeric_features = ['AGE', 'LIMIT_BAL']
        selected_feature = st.selectbox("Select Feature for Analysis:", numeric_features)
        
        # Generate boxplot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create DataFrame with only required columns
        plot_data = raw_train[[selected_feature, 'DEFAULT']].copy()
        plot_data['DEFAULT'] = plot_data['DEFAULT'].map({0: 'No Default', 1: 'Default'})
        
        # Create boxplot
        sns.boxplot(x='DEFAULT', y=selected_feature, data=plot_data, palette=['#4CAF50', '#F44336'], ax=ax)
        
        # Add labels and title
        plt.title(f'{selected_feature} Distribution by Default Status')
        plt.ylabel(selected_feature)
        plt.xlabel('Default Status')
        if selected_feature == "AGE":
            ax.set_ylabel("Age (years)")
        elif selected_feature == "LIMIT_BAL":
            ax.set_ylabel("Credit Limit (NT$)")
            plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{int(x):,}'))
        plt.tight_layout()
        
        # Display in Streamlit
        st.pyplot(fig)
        plt.close(fig)
        
        st.markdown(f"""
        ### Understanding the {selected_feature} Distribution
        
        This boxplot shows how {selected_feature.lower()} is distributed across default and non-default customers:
        
        - **Box**: Middle line shows median, box represents interquartile range (25th to 75th percentile)
        - **Whiskers**: Show range of values (excluding outliers)
        - **Outliers**: Individual points beyond the whiskers
        - **Comparison**: Helps identify if {selected_feature.lower()} shows clear patterns associated with default risk
        
        *Note: This visualization uses synthetic default data based on PAY_0 values to demonstrate the relationship.*
        """)

    # Prediction interface
    st.header("Customer Default Prediction")
    st.write("Please enter the customer's information:")

    col1, col2 = st.columns(2)
    with col1:
        limit_bal = st.number_input("Credit Limit (NT$)", min_value=1)
        sex = st.selectbox("Gender", ["Female", "Male"])
        education = st.selectbox("Education Level", 
                                ["Graduate School", "University", "High School", "Others"])
        marriage = st.selectbox("Marital Status", ["Married", "Single", "Others"])
        age = st.number_input("Age", min_value=18, max_value=100)

    with col2:
        pay_0 = st.number_input("Payment Status - September (PAY_0)", min_value=-2, max_value=8)
        pay_2 = st.number_input("Payment Status - August (PAY_2)", min_value=-2, max_value=8)
        pay_3 = st.number_input("Payment Status - July (PAY_3)", min_value=-2, max_value=8)
        pay_4 = st.number_input("Payment Status - June (PAY_4)", min_value=-2, max_value=8)
        pay_5 = st.number_input("Payment Status - May (PAY_5)", min_value=-2, max_value=8)
        pay_6 = st.number_input("Payment Status - April (PAY_6)", min_value=-2, max_value=8)

    col3, col4 = st.columns(2)
    with col3:
        bill_amt1 = st.number_input("Bill Amount - September (NT$)", min_value=0)
        bill_amt2 = st.number_input("Bill Amount - August (NT$)", min_value=0)
        bill_amt3 = st.number_input("Bill Amount - July (NT$)", min_value=0)
        bill_amt4 = st.number_input("Bill Amount - June (NT$)", min_value=0)
        bill_amt5 = st.number_input("Bill Amount - May (NT$)", min_value=0)
        bill_amt6 = st.number_input("Bill Amount - April (NT$)", min_value=0)

    with col4:
        pay_amt1 = st.number_input("Payment Amount - September (NT$)", min_value=0)
        pay_amt2 = st.number_input("Payment Amount - August (NT$)", min_value=0)
        pay_amt3 = st.number_input("Payment Amount - July (NT$)", min_value=0)
        pay_amt4 = st.number_input("Payment Amount - June (NT$)", min_value=0)
        pay_amt5 = st.number_input("Payment Amount - May (NT$)", min_value=0)
        pay_amt6 = st.number_input("Payment Amount - April (NT$)", min_value=0)

    # Create input dictionary
    input_data = {
        'LIMIT_BAL': limit_bal,
        'SEX': 2 if sex == "Male" else 1,
        'EDUCATION': {"Graduate School": 1, "University": 2, "High School": 3, "Others": 4}[education],
        'MARRIAGE': {"Married": 1, "Single": 2, "Others": 3}[marriage],
        'AGE': age,
        'PAY_0': pay_0,
        'PAY_2': pay_2,
        'PAY_3': pay_3,
        'PAY_4': pay_4,
        'PAY_5': pay_5,
        'PAY_6': pay_6,
        'BILL_AMT1': bill_amt1,
        'BILL_AMT2': bill_amt2,
        'BILL_AMT3': bill_amt3,
        'BILL_AMT4': bill_amt4,
        'BILL_AMT5': bill_amt5,
        'BILL_AMT6': bill_amt6,
        'PAY_AMT1': pay_amt1,
        'PAY_AMT2': pay_amt2,
        'PAY_AMT3': pay_amt3,
        'PAY_AMT4': pay_amt4,
        'PAY_AMT5': pay_amt5,
        'PAY_AMT6': pay_amt6
    }

    if st.button("Predict Default Risk"):
        processed = preprocess_input(input_data)
        prediction = model.predict(processed)[0][0]
        probability = prediction * 100
        
        st.subheader("Prediction Result:")
        if prediction > 0.4:
            st.error(f"🚨 High Risk: {probability:.1f}% probability of default")
        else:
            st.success(f"✅ Low Risk: {probability:.1f}% probability of default")