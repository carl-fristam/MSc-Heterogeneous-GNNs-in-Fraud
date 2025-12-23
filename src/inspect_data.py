import pandas as pd

def quick_inspect(file_path):
    print(f"\n--- Checking {file_path} ---")
    # Load a chunk
    df = pd.read_csv(file_path)
    
    print(f"Total rows (sample): {len(df)}")
    if 'HI-Small_Trans.csv' in file_path:
        print(f"Target distribution:\n{df['Is Laundering'].value_counts(normalize=True)}")
    
    return df.columns

if __name__ == "__main__":
    cols1 = quick_inspect('data/HI-Small_accounts.csv')
    cols2 = quick_inspect('data/HI-Small_Trans.csv')

    print(f"\nAvailable Features (accounts): {list(cols1)} \nAvailable Features (transactions): {list(cols2)}")