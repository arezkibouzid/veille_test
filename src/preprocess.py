import pandas as pd
from sklearn.model_selection import train_test_split
from config.config import DATA_PATH, CANONICAL_LABELS, RANDOM_STATE

def load_and_preprocess_data():
    df = pd.read_csv(DATA_PATH, dtype={'halId_s': str})
    text_columns = ['title_s', 'abstract', 'keywords']
    
    df['text_for_classification'] = (
        df[text_columns]
        .fillna('')
        .astype(str)
        .agg(' '.join, axis=1)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )

    df = df[df['manual_label'].isin(CANONICAL_LABELS)].reset_index(drop=True)

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df['manual_label'],
        random_state=RANDOM_STATE
    )
    return df, train_df, test_df, text_columns

if __name__ == "__main__":
    df, train_df, test_df, _ = load_and_preprocess_data()
    print(f"Total: {len(df)} | Train: {len(train_df)} | Test: {len(test_df)}")
