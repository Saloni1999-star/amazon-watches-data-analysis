# Amazon-watches-data-analysis
An analysis of Amazon smartwatch market trends using Kaggle data
# ⌚ Amazon Smartwatch Market Analysis

## Project Overview

This project aims to provide a comprehensive analysis of the smartwatch market based on product listings scraped from Amazon. Through data cleaning, exploratory data analysis (EDA), and predictive modeling, we uncover key market trends, brand positioning, feature impacts on pricing, and identify challenges in predicting product prices from raw listing data.

## 📊 Dataset

The analysis is based on the `smart_watches_amazon.csv` dataset. This dataset contains information about various smartwatches, including:
* `Name`: Product name
* `Brand`: Smartwatch brand
* `Price`: Listing price (in INR)
* `Screen Size`: Display size (in inches)
* `Special Feature`: Text field describing unique functionalities
* `Model Name`: Specific model identifier
* `Style`: Product style or variant (e.g., "GPS", "GPS + Cellular")
* `Colour`: Product color

## 🚀 Project Phases & Methodology

The project followed a structured data science workflow:

### 1. Data Ingestion & Initial Exploration (`1_data_ingestion.ipynb`)

* **Objective:** Load the raw data and perform preliminary checks.
* **Activities:**
    * Loaded `smart_watches_amazon.csv` into a Pandas DataFrame.
    * Inspected data types, non-null counts, and head of the DataFrame to understand initial structure.

### 2. Data Cleaning & Preprocessing (`2_data_cleaning.ipynb`)

* **Objective:** Transform raw data into a clean, analysis-ready format.
* **Key Steps:**
    * **`price` Column:** Removed currency symbols (₹), commas, and converted to `float64`. Rows with missing prices were dropped.
    * **`Screen Size` Column:** Extracted numerical values (inches) and converted to `float64`. Missing values were imputed with the median `Screen Size`.
    * **High Missing Value Columns:** Columns with a high percentage of missing data (e.g., `Item Weight`, `Battery Life`, `Target Audience`, `Series`, `Age Range (Description)`, `Shape`, `Item Dimensions LxWxH`) were dropped to ensure data quality and relevance.
    * **Categorical Imputation:** Missing values in `Model Name` and `Special Feature` were replaced with 'Unknown' and 'Not Specified', respectively.
    * **Feature Engineering from `Special Feature`:** The `Special Feature` text was analyzed to extract binary flags for common features (e.g., `has_heart_rate_monitor`, `has_gps`, `has_bluetooth_calling`, `has_sleep_monitor`, `has_blood_oxygen_sensor`, `has_water_resistant`, `has_touchscreen`, `has_notifications`, `has_calculator`, `has_altimeter`, `has_barometer`, `has_compass`, `has_music_player`, `has_distance_tracker`, `has_blood_pressure_monitor`).

* **Result:** A clean DataFrame (`df_cleaned`) consisting of 2168 entries and 8 relevant columns (plus new engineered features).

### 3. Exploratory Data Analysis (EDA) (`3_exploratory_data_analysis.ipynb`)

* **Objective:** Uncover patterns, distributions, and initial insights from the cleaned data.
* **Key Insights:**
    * **Brand Dominance:** **Fire-Boltt, Noise, boAt, Crossbeats, and Fastrack** hold the largest market share by listing volume on Amazon, pointing to a strong presence of affordable and local brands.
    * **Price Segmentation:** The market exhibits a clear dual-tier structure:
        * The majority of smartwatches fall within the highly competitive **₹1,000 - ₹2,000** price range.
        * A premium segment exists, with brands like **Apple, Samsung, and Fitbit** offering devices priced **above ₹8,000**, with some reaching significantly higher values.
    * **Standard Screen Sizes:** Most smartwatches feature screen sizes consistently between **1 to 2 inches**.

### 4. Brand & Feature Impact Analysis (`4_brand_feature_analysis.ipynb`)

* **Objective:** Delve deeper into how brands and specific features influence pricing.
* **Key Insights:**
    * **Brand vs. Price:** Premium brands like **Apple, Samsung, and Fitbit** consistently have higher average prices. Conversely, high-volume brands like **Fire-Boltt, Noise, and boAt** are generally found in the more affordable price segments.
    * **Screen Size vs. Price:** Interestingly, `Screen Size` showed **virtually no linear correlation (-0.01)** with `Price`, suggesting that size alone is not a primary cost driver.
    * **Feature-Driven Premiums:** Advanced functionalities significantly increase price. Smartwatches with **GPS** or a **Heart Rate Monitor** command substantially higher average prices (e.g., GPS-enabled watches averaged ~₹45,175, and Heart Rate Monitor-equipped watches averaged ~₹36,003), indicating a clear premium for health, fitness, and navigation capabilities.

### 5. Predictive Modelling (Attempt) (`5_predictive_modelling.ipynb`)

* **Objective:** Build a regression model to predict smartwatch prices based on various features.
* **Methodology:**
    * **Target:** `price`
    * **Features:** `Screen Size`, engineered `has_` features (e.g., `has_gps`), and one-hot encoded categorical features (`Brand`, `Style`, `Colour`, `Model Name`).
    * **Model:** Random Forest Regressor.
    * **Data Split:** 80% training, 20% testing.

* **Initial Results:**
    * **Mean Absolute Error (MAE): 250.53**
    * **R-squared (R2 Score): 1.00**
    * **Top Feature Importances:** `Style_GPS + CELLULAR`, `has_distance_tracker`, `Style_GPS`.

* **Challenge: Persistent Data Leakage:**
    * An R-squared score of **1.00 for real-world data is highly indicative of data leakage**. This means the model inadvertently learned a direct link between some features and the target variable (price), causing it to seemingly predict prices perfectly. The scatter plot of actual vs. predicted prices showed all points perfectly aligned on the diagonal.
    * Despite removing `Model Name` and `Style` (which were strong candidates for leakage due to their high cardinality and potential direct mapping to specific prices), the R2 score remained 1.00 with a non-zero MAE (e.g., 396.36 in later runs). This contradiction suggests a deeper, more subtle leakage or a unique structure within the dataset where certain feature combinations uniquely identify prices.

* **Conclusion on Predictive Model:** While the model achieved a "perfect" R2, this performance is not realistic. It serves as a crucial learning point regarding data leakage in machine learning projects. A truly robust predictive model would require further investigation into the dataset's structure to eliminate all implicit price identifiers or to re-frame the problem (e.g., predicting price ranges or categories).

## 🛠️ Technologies & Libraries

* **Python 3.x**
* **Pandas:** For data manipulation and analysis.
* **NumPy:** For numerical operations.
* **Matplotlib:** For static, animated, and interactive visualizations.
* **Seaborn:** For attractive and informative statistical graphics.
* **Scikit-learn:** For machine learning tasks (model selection, training, evaluation, preprocessing).

## 🚀 How to Run the Project

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
    *(Note: Replace `<repository_url>` and `<repository_name>` with your actual repository details if hosted on GitHub/GitLab)*

2.  **Set up Environment (Recommended: Anaconda/Miniconda):**
    ```bash
    conda create -n smartwatch_env python=3.9
    conda activate smartwatch_env
    ```

3.  **Install Dependencies:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```

4.  **Project Structure:**
    * `data/`:
        * `raw/`: Contains the original `smart_watches_amazon.csv` dataset.
        * `processed/`: (Will store `df_cleaned.csv` if you choose to save it after cleaning).
    * `notebooks/`:
        * `1_data_ingestion.ipynb`
        * `2_data_cleaning.ipynb`
        * `3_exploratory_data_analysis.ipynb`
        * `4_brand_feature_analysis.ipynb`
        * `5_predictive_modelling.ipynb`

5.  **Run Jupyter Notebooks:**
    Navigate to the `notebooks` directory and start Jupyter Lab or Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
    Open the notebooks in sequential order (`1_data_ingestion.ipynb` to `5_predictive_modelling.ipynb`) and run the cells to reproduce the analysis.

## 💡 Future Work

* **Advanced Feature Engineering:** Explore techniques like TF-IDF or word embeddings on `Special Feature` or `Name` for more nuanced text analysis, carefully handling potential leakage.
* **Robust Outlier Handling:** Implement more sophisticated outlier detection and treatment, or use models less sensitive to extreme values.
* **Alternative Modeling:**
    * Experiment with different regression models (e.g., XGBoost, LightGBM) or neural networks.
    * Consider **predicting price ranges (classification problem)** instead of exact prices, which might be more robust to the dataset's inherent structure.
* **Data Source Investigation:** If data leakage persists, a deeper investigation into the original data collection method and unique identifiers might be necessary.
* **Web Scraping Enhancement:** Expand the dataset with more attributes (e.g., ratings, reviews, seller information) for a richer analysis.
