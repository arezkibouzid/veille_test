from src.preprocess import load_and_preprocess_data
from src.encode import load_encoder, get_or_compute_embeddings
from src.train import train_classifier
from src.build_subaxes import compute_subaxis_references
from src.evaluate_and_deploy import evaluate_and_register
from src.predict import predict_article

def main():
    df, train_df, test_df, text_columns = load_and_preprocess_data()
    encoder = load_encoder()
    
    train_embeds, test_embeds, cache_hit, embed_time = get_or_compute_embeddings(encoder, train_df, test_df)
    classifier, mode, fit_time = train_classifier(train_embeds, train_df['manual_label'])
    
    subaxis_refs = compute_subaxis_references(encoder)
    
    evaluate_and_register(
        classifier, encoder, test_embeds, test_df, len(df), 
        text_columns, cache_hit, embed_time, fit_time, mode
    )

    # Sanity-check sample prediction
    res = predict_article(
        encoder, classifier, subaxis_refs,
        title='AI-based intrusion detection for secure networks',
        abstract='We study anomaly detection and cyber defense using machine learning.',
        keywords='cybersecurity, intrusion detection, anomaly detection'
    )
    print("\nInference Output Test:", res)

if __name__ == "__main__":
    main()
