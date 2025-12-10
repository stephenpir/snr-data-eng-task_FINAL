import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get the directory of the current script to build robust file paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
#print(f"Script directory: {SCRIPT_DIR}")
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
#print(f"Project root directory: {PROJECT_ROOT}")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
#print(f"Artifacts directory: {ARTIFACTS_DIR}")

# --- 0. Setup ---
# Create an artifacts directory to save outputs
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# --- 1. Load and Explore Data ---

def load_data(transactions_path, labels_path):
    """Loads transaction and label data from CSV files."""
    print("Loading data...")
    try:
        transactions_df = pd.read_csv(transactions_path)
        labels_df = pd.read_csv(labels_path)
        print("Data loaded successfully.")
        return transactions_df, labels_df
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Please check file paths.")
        return None, None

def explore_data(df, df_name):
    """Performs a basic data quality check and exploration."""
    print(f"\n--- Exploring {df_name} ---")
    print("Shape:", df.shape)
    print("\nInfo:")
    df.info()
    
    # Data Quality Issue: Check for nulls
    null_counts = df.isnull().sum()
    print("\nNull Values:")
    print(null_counts[null_counts > 0])
    if null_counts.sum() == 0:
        print("No null values found.")
   
    # Duplicates assumptions: 
    # For transactions; 
    # - We assume transaction_id should be unique. 
    # - We assume rows are legitimate separate events unless all columns including timestamp are identical. 
    #   i.e. complete duplicate rows can be dropped. As there are no duplicates and it isn't requested in the 
    #   instructions the actual clean up is not implemented.
    # For labels; 
    # - We assume duplicate rows are redundant and can be dropped. 
    # - In the case of multiple rows for the same customer_id, we would need more info to determine choice but 
    #   as there are no duplicates and it isn't requested in the instructions the actual clean up is not implemented.

    # Data Quality Issue: Check for full row duplicates
    num_duplicates = df.duplicated().sum()
    print(f"\nDuplicate Rows: {num_duplicates}")

    # Data Quality Issue: Check for duplicate transaction_id in transactions
    if 'transaction_id' in df.columns and df_name == "Transactions": 
        num_dup_txn_ids = df.duplicated(subset=['transaction_id']).sum()
        print(f"Duplicate Transaction IDs: {num_dup_txn_ids}")
        if num_dup_txn_ids > 0:
            print("Warning: Duplicate transaction_ids found. Each transaction should have a unique ID.")
    # Data Quality Issue: Check for duplicate customer_id in labels
    if 'customer_id' in df.columns and df_name == "Labels": 
        num_dup_cust_ids = df.duplicated(subset=['customer_id']).sum()
        print(f"Duplicate Customer IDs in Labels: {num_dup_cust_ids}")
        if num_dup_cust_ids > 0:
            print("Warning: Duplicate customer_ids found in labels. Each customer should have a single label.")

    # Data Quality Issue: Check for contiguous IDs (transaction_id and customer_id). Not strictly necessary but a good data quality check nonetheless.
    def check_contiguous(series, prefix):
        """Helper function to check for contiguous numeric parts of IDs."""
        # Extract numeric part, convert to int, and drop any that don't match
        numeric_ids = pd.to_numeric(series.str.extract(f'^{prefix}(\d+)', expand=False), errors='coerce').dropna().astype(int)

        if numeric_ids.empty:
            print(f"Could not find any valid numeric IDs with prefix '{prefix}' to check for contiguity.")
            return

        sorted_unique_ids = np.sort(numeric_ids.unique())
        min_id, max_id = sorted_unique_ids[0], sorted_unique_ids[-1]
        
        # A sequence is contiguous if the count of unique IDs equals the range span
        if len(sorted_unique_ids) == (max_id - min_id + 1):
            print(f"IDs with prefix '{prefix}' are contiguous from {min_id} to {max_id}.")
        else:
            # Find the missing numbers for a more detailed warning
            expected_range = set(range(min_id, max_id + 1))
            missing_ids = sorted(list(expected_range - set(sorted_unique_ids)))
            print(f"Warning: IDs with prefix '{prefix}' are not contiguous. Found {len(missing_ids)} missing values in the sequence.")

    if 'transaction_id' in df.columns:
        check_contiguous(df['transaction_id'], 'T')
    if 'customer_id' in df.columns:
        check_contiguous(df['customer_id'], 'CUST_')

    # Data Quality Issue: Check for outliers (in transaction amount)
    # An outlier could be an amount significantly far from the mean/median.
    # We can flag transactions > 3 standard deviations from the mean.
    # I've opted to separate credits and debits for outlier detection.
    if 'amount' in df.columns:
        print("\nTransaction Amount Description:")
        print(df['amount'].describe())

        # Actual test for outliers using the 3-sigma rule, split by credit and debit
        credits = df[df['amount'] > 0]
        debits = df[df['amount'] < 0]

        # --- Check Credit Outliers (unusually large credits) ---
        if not credits.empty:
            mean_credit = credits['amount'].mean()
            std_credit = credits['amount'].std()
            upper_bound_credit = mean_credit + 3 * std_credit
            credit_outliers = credits[credits['amount'] > upper_bound_credit]

            if not credit_outliers.empty:
                print(f"\nWarning: Found {len(credit_outliers)} potential outliers in credit transactions.")
                print("A few examples of unusually large credits:")
                print(credit_outliers.head())
            else:
                print("\nNo significant outliers found in credit transactions.")

        # --- Check Debit Outliers (unusually large debits) ---
        if not debits.empty:
            mean_debit = debits['amount'].mean()
            std_debit = debits['amount'].std()
            lower_bound_debit = mean_debit - 3 * std_debit # Note: For debits, this will be a more negative number
            debit_outliers = debits[debits['amount'] < lower_bound_debit]

            if not debit_outliers.empty:
                print(f"\nWarning: Found {len(debit_outliers)} potential outliers in debit transactions.")
                print("A few examples of unusually large debits:")
                print(debit_outliers.head())
            else:
                print("No significant outliers found in debit transactions.")


# --- 2. Feature Engineering ---

def engineer_numerical_features(df):
    """Aggregates transactions and creates numerical features for each customer."""
    print("\nEngineering numerical features...")
    
    # Ensure amount is numeric
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

    # Aggregate all required features in a single groupby operation
    features = df.groupby('customer_id').agg(
        num_transactions=('transaction_id', 'count'),
        avg_transaction_amount=('amount', 'mean'),
        total_debit=('amount', lambda x: x[x < 0].sum()),
        total_credit=('amount', lambda x: x[x > 0].sum())
    ).reset_index()

    # My own 3 features
    # Feature 1: Net balance change
    features['balance_change'] = features['total_credit'] + features['total_debit']

    # Feature 2: Ratio of total debit amount to total credit amount
    features['debit_credit_amount_ratio'] = abs(features['total_debit']) / (features['total_credit'] + 1) # Add 1 to avoid division by zero

    # Feature 3: Standard deviation of transaction amounts as a measure of volatility
    amount_std = df.groupby('customer_id').agg(
        std_dev_amount=('amount', 'std')
    ).reset_index().fillna(0) # Fillna for customers with only one transaction
    features = pd.merge(features, amount_std, on='customer_id')

    print("Numerical features created.")
    return features


# --- 3. Text Processing on Description ---

def engineer_text_features(df):
    """Cleans description text and extracts keyword-based features."""
    print("\nEngineering text features from description...")

    # Clean the text: lowercase, remove punctuation and numbers
    df['description_clean'] = (df['description'].str
                               .lower()  # a. Convert all text to lowercase.
                               .str.replace(r'[^a-z\s]', '', regex=True)  # b. Remove characters that are not letters or whitespace.
                               .str.strip())  # c. Remove any leading or trailing whitespace.

    # Define keywords for common categories
    keywords = {
        'has_rent': 'rent',
        'has_payroll': 'payroll|upwork|payout', # Added 'upwork' and 'payout' as possible payroll indicators
        'has_netflix': 'netflix',
        'has_tesco': 'tesco',
        'has_bonus': 'bonus'
    }

    text_features = pd.DataFrame(index=df['customer_id'].unique())

    for feature_name, keyword_regex in keywords.items():
        # For each customer, check if any of their transactions match the keyword
        customer_flags = df.groupby('customer_id')['description_clean'].apply(
            lambda descriptions: 1 if descriptions.str.contains(keyword_regex).any() else 0
        )
        text_features[feature_name] = customer_flags

    print("Text features created.")
    return text_features.reset_index().rename(columns={'index': 'customer_id'})


# --- 4. Create Training Dataset ---

def create_training_set(numerical_features, text_features, labels_df):
    """Merges all features with labels to create the final training set."""
    print("\nCreating final training set...")
    
    # Merge numerical and text features
    training_df = pd.merge(numerical_features, text_features, on='customer_id')
    
    # Merge with labels
    training_df = pd.merge(training_df, labels_df, on='customer_id')

    # Final check for duplicates on customer_id. We assume duplicates should be dropped keeping the first occurrence. 
    # This would not be necessary if data quality checks and clensing were implemented earlier.
    training_df = training_df.drop_duplicates(subset=['customer_id'], keep='first')

    print("Training set created successfully.")
    return training_df


# --- 5. Exploratory Data Analysis ---

def run_eda(df):
    """Generates and saves key visualizations from the training data."""
    print("\nRunning Exploratory Data Analysis...")

    # Plot 1: Histogram of Number of Transactions per Customer
    plt.figure(figsize=(10, 6))
    sns.histplot(df['num_transactions'], kde=False, bins=max(1, df['num_transactions'].max())) # Bins based on actual data
    plt.title('Distribution of Number of Transactions per Customer')
    plt.xlabel('Number of Transactions')
    plt.ylabel('Frequency (Number of Customers)')
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'num_transactions_dist.png'))
    print(f"Saved plot: {os.path.join(ARTIFACTS_DIR, 'num_transactions_dist.png')}")

    # Plot 2: Histogram of Average Transaction Amount
    plt.figure(figsize=(10, 6))
    sns.histplot(df['avg_transaction_amount'], kde=True, bins=20)
    plt.title('Distribution of Average Transaction Amount per Customer')
    plt.xlabel('Average Transaction Amount')
    plt.ylabel('Frequency (Number of Customers)')
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'avg_txn_amount_dist.png'))
    print(f"Saved plot: {os.path.join(ARTIFACTS_DIR, 'avg_txn_amount_dist.png')}")

    # Plot 3: Scatter plot of Total Credit vs Total Debit, coloured by default status
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df, x='total_debit', y='total_credit', hue='defaulted_within_90d', alpha=0.8)
    plt.title('Total Credit vs. Total Debit by Default Status')
    plt.xlabel('Total Debit (Negative)')
    plt.ylabel('Total Credit (Positive)')
    plt.grid(True)
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'credit_vs_debit_scatter.png'))
    print(f"Saved plot: {os.path.join(ARTIFACTS_DIR, 'credit_vs_debit_scatter.png')}")

    plt.close('all') # Close all figures


if __name__ == '__main__':
    # Define file paths
    TRANSACTIONS_FILE = os.path.join(PROJECT_ROOT, 'data', 'transactions.csv')
    LABELS_FILE = os.path.join(PROJECT_ROOT, 'data', 'labels.csv')
    OUTPUT_FILE = os.path.join(ARTIFACTS_DIR, 'training_set.csv')

    # 1. Load and explore
    transactions, labels = load_data(TRANSACTIONS_FILE, LABELS_FILE)
    if transactions is not None and labels is not None:
        explore_data(transactions, "Transactions")
        explore_data(labels, "Labels")

        # 2. Engineer numerical features
        numerical_features_df = engineer_numerical_features(transactions.copy())

        # 3. Engineer text features
        text_features_df = engineer_text_features(transactions.copy())

        # 4. Create training set
        training_dataset = create_training_set(numerical_features_df, text_features_df, labels)
        
        # Save the result
        training_dataset.to_csv(OUTPUT_FILE, index=False)
        print(f"\nFinal training set saved to {OUTPUT_FILE}")
        print("Final dataset preview:")
        print(training_dataset.head())

        # 5. Run EDA
        run_eda(training_dataset)
