import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from PIL import Image

# Add some space
st.sidebar.write(" ")

# Add contributors section with links to their GitHub profiles
st.sidebar.markdown("### Creator:")
st.sidebar.markdown("""
- [Mehdi Seyedebrahimi (git URL)](https://github.com/mirmehdi)
""")

# Add client Section
st.sidebar.markdown("### Interactive Interface _ Data Analytics:")
st.sidebar.markdown("""
- [Berlin Business School and Innovation GmbH ](https://www.berlinsbi.com/)
""")
current_dir = os.path.abspath(os.path.dirname(__file__))

image_path_1 = os.path.join(current_dir, 'logo.png')
image = Image.open(image_path_1) 
st.markdown(f"""
                <img src="data:image/png;base64,{st.image(image, caption=None, use_column_width=False, width=670)}">
            """, unsafe_allow_html=True)

st.title("Dimensionality Reduction with PCA")

# Step 1: Upload CSV File
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Step 2: Load the dataset
    df = pd.read_csv(uploaded_file, sep='\t')
    df.columns = df.columns.str.strip()

    # Select relevant features and target variable
    features = ['Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 
                'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 
                'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 
                'NumStorePurchases', 'NumWebVisitsMonth']
    target = 'Response'

    df_subset = df[features].dropna()
    df_response = df.loc[df_subset.index, target]

    st.title("Data source summary")
    # Display data matrix and summary
    st.write("DataFrame Summary:")
    st.write("Data: 13 features (info) about 2500 customers are used to predict if they will respond positively to our product or not.")
    st.write(df_subset.head(10))
    st.write("Features:")
    buffer = pd.DataFrame({
        'Columns': df_subset.columns,
    })
    st.write(buffer)

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_subset)

    # Perform PCA
    pca = PCA()
    pca.fit(scaled_data)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)

    st.title("Eigenvalue plot")
    # Plot the eigenvalues (scree plot)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='--')
    ax.set_title('Scree Plot')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Eigenvalue')
    st.pyplot(fig)

    st.title("Cumulative explained variance")
    # Plot the cumulative explained variance
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
    ax.axhline(y=0.90, color='r', linestyle='-')
    ax.set_title('Cumulative Explained Variance')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance')
    st.pyplot(fig)

    st.title("Choose the number of PC components")
    # Step 4: Choose the number of principal components
    max_components = len(cumulative_explained_variance)
    initial_value = min(max_components, int(np.argmax(cumulative_explained_variance >= 0.90) + 1))
    num_components = st.slider("Select number of principal components", 
                               min_value=1, 
                               max_value=max_components, 
                               value=initial_value)
    st.write(f"Number of components selected: {num_components}")

    # Transform the data using the selected number of components
    pca = PCA(n_components=num_components)
    principal_components = pca.fit_transform(scaled_data)

    # Create a DataFrame with the principal components
    df_principal_components = pd.DataFrame(data=principal_components, 
                                           columns=[f'PC{i+1}' for i in range(num_components)])
    df_principal_components['Response'] = df_response.values

    # Split the data into training and testing sets for PCA-transformed data
    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(df_principal_components.drop('Response', axis=1), 
                                                                        df_principal_components['Response'], 
                                                                        test_size=0.2, random_state=42)

    # Split the data into training and testing sets for original features
    X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(df_subset, 
                                                                        df_response, 
                                                                        test_size=0.2, random_state=42)

    # Train and evaluate Random Forest Classifier with PCA features
    model_pca = RandomForestClassifier(n_estimators=100, random_state=42)
    model_pca.fit(X_train_pca, y_train_pca)
    y_pred_pca = model_pca.predict(X_test_pca)
    accuracy_pca = accuracy_score(y_test_pca, y_pred_pca)
    report_pca = classification_report(y_test_pca, y_pred_pca, output_dict=True)
    st.title("Result of ML model after PCA and dimensionality reduction")
    st.write("Model with PCA features")
    st.write(f"Accuracy: {accuracy_pca:.2f}")
    #st.write(pd.DataFrame(report_pca).transpose())

    # Train and evaluate Random Forest Classifier with all features
    model_all = RandomForestClassifier(n_estimators=100, random_state=42)
    model_all.fit(X_train_all, y_train_all)
    y_pred_all = model_all.predict(X_test_all)
    accuracy_all = accuracy_score(y_test_all, y_pred_all)
    report_all = classification_report(y_test_all, y_pred_all, output_dict=True)

    st.title("Result of ML model using raw data")
    st.write("Model with all features")
    st.write(f"Accuracy: {accuracy_all:.2f}")
    #st.write(pd.DataFrame(report_all).transpose())
