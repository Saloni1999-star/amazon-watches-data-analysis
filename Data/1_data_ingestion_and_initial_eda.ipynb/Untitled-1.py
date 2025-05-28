# %%
# Import essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: display settings for pandas and seaborn
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")


# %%
import pandas as pd

# Load the dataset
file_path = 'smart_watches_amazon.csv.csv'
df = pd.read_csv(file_path)

# %%
# Display the first few rows
print("\nFirst 5 rows of the dataset:")
df.head()

# %%
#Get a summary of the DataFrame
print("\nDataFrame Info (non-nulls and data types):")
df.info()

# %%
# Get descriptive statistics for numerical columns
print("\nDescriptive Statistics for numerical columns:")
df.describe()

# %%
# Check the column names
print("\nColumn Names:")
print(df.columns)

# %%
# Cell 7: Check the shape (number of rows and columns)
print(f"\nDataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

# %%
# Create a copy of the original DataFrame to work on, good practice
df_cleaned = df.copy()

# %%
# --- Cleaning 'price' Column ---

# 1. Inspect unique values and identify non-numeric characters (optional, but good for understanding)
# This will show you what kind of strings are in the price column that prevent direct conversion
# print("Unique price values (first 20, to spot patterns):")
# print(df_cleaned['price'].unique()[:20]) # Look for symbols, commas, ranges

# %%
# 2. Remove currency symbols, commas, and other non-numeric characters
# Using .str.replace() to handle potential NaN values by chaining .fillna('')
df_cleaned['price'] = df_cleaned['price'].astype(str).str.replace('₹', '', regex=False) # Remove Indian Rupee symbol
df_cleaned['price'] = df_cleaned['price'].str.replace('$', '', regex=False)  # Remove Dollar symbol
df_cleaned['price'] = df_cleaned['price'].str.replace(',', '', regex=False)  # Remove commas
df_cleaned['price'] = df_cleaned['price'].str.replace(' ', '', regex=False)  # Remove spaces
# Add more .replace() calls if you find other symbols or text (e.g., 'From ', 'Up to ')

# %%
# 3. Convert to numeric (float)
# Use errors='coerce' to turn values that cannot be converted into NaN
df_cleaned['price'] = pd.to_numeric(df_cleaned['price'], errors='coerce')

# %%
# 4. Handle missing values in 'price' after conversion
# We saw ~148 missing values from df.info(). After coercing errors, there might be more.
print(f"Missing values in 'price' column before handling: {df_cleaned['price'].isnull().sum()}")

# %%
# For price, dropping rows with missing values is generally a safe approach to avoid distortion.
# If you wanted to impute, you might use mean/median, but dropping is often preferred for financial data.
df_cleaned.dropna(subset=['price'], inplace=True) # inplace=True modifies the DataFrame directly

print(f"Missing values in 'price' column after dropping: {df_cleaned['price'].isnull().sum()}")
print(f"DataFrame shape after dropping rows with missing price: {df_cleaned.shape}")

# %%
# 5. Verify the data type
print("\n'price' column info after cleaning:")
df_cleaned['price'].info()
print(df_cleaned['price'].head())

# %%
# --- Cleaning 'Screen Size' Column ---

# 1. Inspect unique values to identify common patterns (units, text)
print("Unique 'Screen Size' values (first 20, to spot patterns):")
# Convert to string to avoid errors with NaN if inspecting directly
print(df_cleaned['Screen Size'].astype(str).unique()[:20])

# %%
# 2. Remove common units and other non-numeric characters
# Using .str.replace() for string operations
df_cleaned['Screen Size'] = df_cleaned['Screen Size'].astype(str).str.replace('Inches', '', regex=False)
df_cleaned['Screen Size'] = df_cleaned['Screen Size'].str.replace('inch', '', regex=False)
df_cleaned['Screen Size'] = df_cleaned['Screen Size'].str.replace('"', '', regex=False) # Remove double quotes
df_cleaned['Screen Size'] = df_cleaned['Screen Size'].str.strip() # Remove leading/trailing whitespace


# %%
# 3. Convert to numeric (float)
# Use errors='coerce' to turn values that cannot be converted into NaN
df_cleaned['Screen Size'] = pd.to_numeric(df_cleaned['Screen Size'], errors='coerce')

# %%
# 4. Handle missing values in 'Screen Size' after conversion
print(f"\nMissing values in 'Screen Size' column before handling: {df_cleaned['Screen Size'].isnull().sum()}")

# For 'Screen Size', imputing with the median might be a reasonable strategy
# if the distribution is skewed, or mean if it's normal.
# Let's use the median for now, as screen sizes can vary significantly.
median_screen_size = df_cleaned['Screen Size'].median()
df_cleaned['Screen Size'] = df_cleaned['Screen Size'].fillna(median_screen_size)

print(f"Missing values in 'Screen Size' column after imputation: {df_cleaned['Screen Size'].isnull().sum()}")

# %%
# 5. Verify the data type and look at cleaned head
print("\n'Screen Size' column info after cleaning:")
df_cleaned['Screen Size'].info()
print(df_cleaned['Screen Size'].head())

# %%
# --- Cleaning 'Item Weight' Column ---

# 1. Inspect unique values to identify common patterns (units, text)
print("\nUnique 'Item Weight' values (first 20, to spot patterns):")
print(df_cleaned['Item Weight'].astype(str).unique()[:20])

# %%
# 2. all text to lowercase for easier replacement
df_cleaned['Item Weight'] = df_cleaned['Item Weight'].astype(str).str.lower()

# %%
# 3. Remove common units and other non-numeric characters
# Order of replacement matters: replace 'kg' before 'g', 'lbs' before 'lb'
df_cleaned['Item Weight'] = df_cleaned['Item Weight'].str.replace('kilograms', '', regex=False)
df_cleaned['Item Weight'] = df_cleaned['Item Weight'].str.replace('kg', '', regex=False)
df_cleaned['Item Weight'] = df_cleaned['Item Weight'].str.replace('grams', '', regex=False)
df_cleaned['Item Weight'] = df_cleaned['Item Weight'].str.replace('g', '', regex=False)
df_cleaned['Item Weight'] = df_cleaned['Item Weight'].str.replace('pounds', '', regex=False)
df_cleaned['Item Weight'] = df_cleaned['Item Weight'].str.replace('lbs', '', regex=False)
df_cleaned['Item Weight'] = df_cleaned['Item Weight'].str.replace('lb', '', regex=False)
df_cleaned['Item Weight'] = df_cleaned['Item Weight'].str.replace('ounces', '', regex=False)
df_cleaned['Item Weight'] = df_cleaned['Item Weight'].str.replace('oz', '', regex=False)
df_cleaned['Item Weight'] = df_cleaned['Item Weight'].str.replace(';', '', regex=False) # Remove semicolons if any
df_cleaned['Item Weight'] = df_cleaned['Item Weight'].str.strip() # Remove leading/trailing whitespace


# %%
# 4. Convert to numeric (float)
# Use errors='coerce' to turn values that cannot be converted into NaN
df_cleaned['Item Weight'] = pd.to_numeric(df_cleaned['Item Weight'], errors='coerce')

# %%
# 5. Handle missing values in 'Item Weight' after conversion
# Given the very high number of missing values,
# instead of imputing, let's fill with 0 or a specific placeholder to indicate 'unknown/not specified'.
# We can also decide to drop the column later if it doesn't add value.
print(f"\nMissing values in 'Item Weight' column before handling: {df_cleaned['Item Weight'].isnull().sum()}")

# Filling with 0 implies no weight, which might be misleading.
# Let's fill with a placeholder (e.g., -1 or median for now) and note its sparsity.
# For now, let's fill with median, similar to Screen Size, but be aware of its impact later.
# If the goal is to use this in a model, filling with 0 and adding a 'weight_is_missing' flag is common.
median_item_weight = df_cleaned['Item Weight'].median()
df_cleaned['Item Weight'].fillna(median_item_weight, inplace=True) # Or df_cleaned['Item Weight'] = df_cleaned['Item Weight'].fillna(median_item_weight)

print(f"Missing values in 'Item Weight' column after imputation: {df_cleaned['Item Weight'].isnull().sum()}")

# %%
# 6. Verify the data type and look at cleaned head
print("\n'Item Weight' column info after cleaning:")
df_cleaned['Item Weight'].info()
print(df_cleaned['Item Weight'].head())

# %%
# --- Handling Highly Sparse Columns ---

# List of columns to drop due to high percentage of missing values
columns_to_drop = [
    'Target Audience',
    'Series',
    'Age Range (Description)',
    'Shape',
    'Item Dimensions LxWxH',
    'Item Weight',  # Even though we cleaned it, its sparsity limits its utility
    'Battery Life'  # Also extremely sparse
]
print(f"\nDataFrame shape before dropping sparse columns: {df_cleaned.shape}")
df_cleaned.drop(columns=columns_to_drop, inplace=True, errors='ignore')
print(f"DataFrame shape after dropping sparse columns: {df_cleaned.shape}")

# Verify the columns are dropped
print("\nColumns after dropping sparse ones:")
print(df_cleaned.columns)


# %%
# --- Handling Missing Values in other Categorical Columns ---

# Fill missing 'Model Name' with 'Unknown'
print(f"\nMissing values in 'Model Name' before handling: {df_cleaned['Model Name'].isnull().sum()}")
df_cleaned['Model Name'] = df_cleaned['Model Name'].fillna('Unknown')
print(f"Missing values in 'Model Name' after handling: {df_cleaned['Model Name'].isnull().sum()}")

# Fill missing 'Special Feature' with 'Not Specified' or 'No Special Feature'
print(f"\nMissing values in 'Special Feature' before handling: {df_cleaned['Special Feature'].isnull().sum()}")
df_cleaned['Special Feature'] = df_cleaned['Special Feature'].fillna('Not Specified')
print(f"Missing values in 'Special Feature' after handling: {df_cleaned['Special Feature'].isnull().sum()}")

# Verify all changes by checking df_cleaned.info() again
print("\nFinal DataFrame Info after all cleaning steps:")
df_cleaned.info()

# %%
# --- Exploratory Data Analysis (EDA) ---

# Get descriptive statistics for numerical columns
print("\nDescriptive Statistics for Numerical Columns:")
print(df_cleaned.describe())

# Get descriptive statistics for all columns, including categorical ones
print("\nDescriptive Statistics for All Columns (including categorical):")
print(df_cleaned.describe(include='all'))

# %%
# --- Top Brands Analysis ---

# Get the count of each brand
brand_counts = df_cleaned['Brand'].value_counts()
print("\nTop 10 Smartwatch Brands by Count:")
print(brand_counts.head(10))

# Visualize Top 10 Brands
plt.figure(figsize=(12, 6))
sns.barplot(x=brand_counts.head(10).index, y=brand_counts.head(10).values, palette='viridis')
plt.title('Top 10 Smartwatch Brands on Amazon')
plt.xlabel('Brand')
plt.ylabel('Number of Listings')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %%
# --- Price Distribution Analysis ---

# Histogram of prices
plt.figure(figsize=(10, 6))
sns.histplot(df_cleaned['price'], kde=True, bins=30, color='skyblue')
plt.title('Distribution of Smartwatch Prices')
plt.xlabel('Price (₹)')
plt.ylabel('Number of Listings')
plt.tight_layout()
plt.show()

# Box plot of prices to identify outliers
plt.figure(figsize=(8, 6))
sns.boxplot(y=df_cleaned['price'], color='lightcoral')
plt.title('Box Plot of Smartwatch Prices')
plt.ylabel('Price (₹)')
plt.tight_layout()
plt.show()

# %%
# --- Screen Size Distribution Analysis ---

# Histogram of screen sizes
plt.figure(figsize=(10, 6))
sns.histplot(df_cleaned['Screen Size'], kde=True, bins=20, color='lightgreen')
plt.title('Distribution of Smartwatch Screen Sizes')
plt.xlabel('Screen Size (Inches)')
plt.ylabel('Number of Listings')
plt.tight_layout()
plt.show()

# Box plot of screen sizes
plt.figure(figsize=(8, 6))
sns.boxplot(y=df_cleaned['Screen Size'], color='lightsalmon')
plt.title('Box Plot of Smartwatch Screen Sizes')
plt.ylabel('Screen Size (Inches)')
plt.tight_layout()
plt.show()

# %%
# --- Brand vs. Price Analysis ---

# Calculate the average price for each brand
average_price_by_brand = df_cleaned.groupby('Brand')['price'].mean().sort_values(ascending=False)

print("\nAverage Price by Brand (Top 15):")
print(average_price_by_brand.head(15))

print("\nAverage Price by Brand (Bottom 15):")
print(average_price_by_brand.tail(15))

# %%
# Visualize Average Price of Top 15 Brands
plt.figure(figsize=(14, 7))
sns.barplot(x=average_price_by_brand.head(15).index, y=average_price_by_brand.head(15).values, palette='coolwarm')
plt.title('Average Smartwatch Price by Brand (Top 15)')
plt.xlabel('Brand')
plt.ylabel('Average Price (₹)')
plt.xticks(rotation=60, ha='right')
plt.tight_layout()
plt.show()

# %%
# Select top N brands by count for box plot analysis (e.g., top 5)
# Re-using brand_counts from previous EDA or recalculating if needed
brand_counts = df_cleaned['Brand'].value_counts()
top_brands_by_count = brand_counts.head(5).index.tolist()

# Filter the DataFrame for these top brands
df_top_brands = df_cleaned[df_cleaned['Brand'].isin(top_brands_by_count)]

# Create a box plot to show price distribution for top brands
plt.figure(figsize=(14, 8))
sns.boxplot(x='Brand', y='price', data=df_top_brands, palette='pastel')
plt.title('Smartwatch Price Distribution for Top 5 Brands (by Listing Count)')
plt.xlabel('Brand')
plt.ylabel('Price (₹)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %%
# --- Screen Size vs. Price Analysis ---

# Calculate the correlation coefficient
correlation_screen_price = df_cleaned['Screen Size'].corr(df_cleaned['price'])
print(f"\nCorrelation between Screen Size and Price: {correlation_screen_price:.2f}")

# Create a scatter plot to visualize the relationship
plt.figure(figsize=(10, 7))
sns.scatterplot(x='Screen Size', y='price', data=df_cleaned, alpha=0.6, s=50, color='purple')
plt.title('Smartwatch Price vs. Screen Size')
plt.xlabel('Screen Size (Inches)')
plt.ylabel('Price (₹)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# %%
# --- Initial Exploration of 'Special Feature' ---

# Display the value counts for Special Feature (top 20)
# This will show 'Not Specified' as the most common, but also other common combinations
print("\nTop 20 Special Feature Combinations:")
print(df_cleaned['Special Feature'].value_counts().head(20))

# Sample some unique values to understand the variety and common phrasing
print("\nSample of 30 unique Special Feature entries:")
print(df_cleaned['Special Feature'].unique()[:30])

# %%
# --- Feature Impact Analysis: Processing 'Special Feature' ---

# Based on your observations from the 'Special Feature' unique values and counts,
# let's define a list of common features to extract.
# We'll use more general terms where possible (e.g., 'calling' for 'phone call', 'bluetooth calling').
features_keywords = {
    'Heart Rate Monitor': ['heart rate monitor', 'heart rate'],
    'GPS': ['gps'],
    'Bluetooth Calling': ['bluetooth calling', 'phone call', 'calling'],
    'Sleep Monitor': ['sleep monitor', 'sleep tracking'],
    'Activity Tracker': ['activity tracker', 'multisport tracker', 'sports modes'],
    'Blood Oxygen Sensor': ['spo2', 'blood oxygen'],
    'Water Resistant': ['waterproof', 'water resistant'], # Assuming these are common if not explicitly seen
    'Touchscreen': ['touchscreen', 'hd display'], # Assuming HD Display implies touchscreen
    'Notifications': ['notifications'],
    'Calculator': ['calculator'], 'Altimeter': ['altimeter'],
    'Barometer': ['barometer'],
    'Compass': ['compass'],
    'Music Player': ['music player'],
    'Distance Tracker': ['distance tracker'],
    'Blood Pressure Monitor': ['blood pressure monitor']
}

# Convert 'Special Feature' to lowercase to make matching case-insensitive
df_cleaned['Special Feature'] = df_cleaned['Special Feature'].astype(str).str.lower()


# %%
# Create new binary (dummy) columns for each feature
print("\nCreating binary feature columns...")
for feature_name, keywords in features_keywords.items():
    # Create a regex pattern to match any of the keywords
    pattern = '|'.join(keywords)
    df_cleaned[f'has_{feature_name.lower().replace(" ", "_")}'] = df_cleaned['Special Feature'].str.contains(pattern, na=False)

print("Binary feature columns created.")

# %%
# Display the head of the DataFrame with new feature columns to verify
print("\nDataFrame head with new feature columns:")
print(df_cleaned.head())

# %%
# --- Analyze Price Impact of Key Features (Example: Heart Rate Monitor) ---

# Example: Average price for watches with and without Heart Rate Monitor
avg_price_heart_rate = df_cleaned.groupby('has_heart_rate_monitor')['price'].mean()
print(f"\nAverage Price by 'has_heart_rate_monitor':\n{avg_price_heart_rate}")

# Example: Visualize price distribution for watches with/without Heart Rate Monitor
plt.figure(figsize=(10, 6))
sns.boxplot(x='has_heart_rate_monitor', y='price', data=df_cleaned, palette='coolwarm')
plt.title('Price Distribution: Watches With vs. Without Heart Rate Monitor')
plt.xlabel('Has Heart Rate Monitor')
plt.ylabel('Price (₹)')
plt.xticks([0, 1], ['No', 'Yes'])
plt.tight_layout()
plt.show()

# %%
# You can repeat this analysis for other features like 'has_gps', 'has_bluetooth_calling', etc.
# For instance, let's also look at GPS
avg_price_gps = df_cleaned.groupby('has_gps')['price'].mean()
print(f"\nAverage Price by 'has_gps':\n{avg_price_gps}")

plt.figure(figsize=(10, 6))
sns.boxplot(x='has_gps', y='price', data=df_cleaned, palette='coolwarm')
plt.title('Price Distribution: Watches With vs. Without GPS')
plt.xlabel('Has GPS')
plt.ylabel('Price (₹)')
plt.xticks([0, 1], ['No', 'Yes'])
plt.tight_layout()
plt.show()

# %%
# --- Predictive Modelling ---

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Make sure df_cleaned is available (if starting a new notebook, reload or run previous cells)
# If you are in a new notebook, ensure you run the data loading and cleaning cells from previous steps
# For example:
# file_path = '../data/raw/smart_watches_amazon.csv'
# df = pd.read_csv(file_path)
# df_cleaned = df.copy()
# (Then run all the cleaning steps from Phase 2 to get df_cleaned as it was)
# (Also, ensure all the `has_` features are generated as per Feature Impact Analysis section)

# Define target variable
target = 'price'
y = df_cleaned[target]

# Define features (X)
# Select numerical features
numerical_features = ['Screen Size'] + [col for col in df_cleaned.columns if col.startswith('has_')]

# %%
# Select categorical features for one-hot encoding
categorical_features = ['Brand', 'Style', 'Colour', 'Model Name']

# Create dummy variables for categorical features
X_categorical = pd.get_dummies(df_cleaned[categorical_features], drop_first=True) # drop_first avoids multicollinearity

# Combine numerical and one-hot encoded categorical features
X = pd.concat([df_cleaned[numerical_features], X_categorical], axis=1)

print(f"\nShape of features DataFrame (X): {X.shape}")
print("\nFeatures (X) head after encoding:")
print(X.head())

# %%
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nShape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")

# %%
# Step 4: Model Selection (Random Forest Regressor)
# We already imported RandomForestRegressor from sklearn.ensemble earlier.
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1 uses all available cores

print("\nRandom Forest Regressor model initialized.")

# %%
# Step 5: Model Training
print("Training the model...")
model.fit(X_train, y_train)
print("Model training complete.")

# %%
# Step 6: Model Evaluation
print("\nEvaluating the model on the test set...")
y_pred = model.predict(X_test)

# %%
# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R2 Score): {r2:.2f}")

# %%
# Visualize actual vs. predicted prices
plt.figure(figsize=(10, 7))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # Perfect prediction line
plt.title('Actual vs. Predicted Smartwatch Prices')
plt.xlabel('Actual Price (₹)')
plt.ylabel('Predicted Price (₹)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# %%
# Feature Importance: Understand which features the model found most important
print("\nTop 10 Feature Importances:")
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(feature_importances.head(10))


