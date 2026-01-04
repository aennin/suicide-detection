# Step 1: Enhanced Data Loading and Preprocessing
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import warnings
warnings.filterwarnings('ignore')

# Load and clean data
def load_and_clean_data(file_path):
    """Load and preprocess the dataset"""
    df = pd.read_csv(file_path)
    
    # Drop unnamed columns
    df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")], errors="ignore")
    
    # Keep only necessary columns
    text_col = "text"
    label_col = "class"
    df = df[[text_col, label_col]].dropna(subset=[text_col, label_col])
    
    # Clean text
    def clean_text(s):
        s = str(s).lower()
        s = re.sub(r"http\S+|www\.\S+", " ", s)     # URLs
        s = re.sub(r"@\w+|#\w+", " ", s)            # @handles, #tags
        s = re.sub(r"[^a-z0-9\s']", " ", s)         # Keep alphanumeric + apostrophes/spaces
        s = re.sub(r"\s+", " ", s).strip()
        return s
    
    df["text_clean"] = df[text_col].map(clean_text)
    df[label_col] = df[label_col].str.lower().str.strip()
    
    return df, text_col, label_col

# Load data
df, TEXT_COL, LABEL_COL = load_and_clean_data("Suicide_Detection.csv")

# Step 2: Comprehensive Exploratory Data Analysis (EDA)
def perform_eda(df, label_col):
    """Perform comprehensive exploratory data analysis"""
    
    # 1. Class distribution
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    class_counts = df[label_col].value_counts()
    plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Class Distribution')
    
    # 2. Text length distribution
    plt.subplot(2, 3, 2)
    df['text_length'] = df['text_clean'].str.len()
    sns.histplot(data=df, x='text_length', hue=label_col, bins=50)
    plt.title('Text Length Distribution by Class')
    plt.xlabel('Text Length (characters)')
    
    # 3. Word count distribution
    plt.subplot(2, 3, 3)
    df['word_count'] = df['text_clean'].str.split().str.len()
    sns.boxplot(data=df, x=label_col, y='word_count')
    plt.title('Word Count by Class')
    plt.xticks(rotation=45)
    
    # 4. Most common words
    from collections import Counter
    plt.subplot(2, 3, 4)
    suicide_texts = ' '.join(df[df[label_col] == 'suicide']['text_clean'])
    suicide_words = Counter(suicide_texts.split()).most_common(10)
    words, counts = zip(*suicide_words)
    plt.bar(words, counts)
    plt.title('Top 10 Words - Suicide Class')
    plt.xticks(rotation=45)
    
    # 5. Most common words - non-suicide
    plt.subplot(2, 3, 5)
    non_suicide_texts = ' '.join(df[df[label_col] == 'non-suicide']['text_clean'])
    non_suicide_words = Counter(non_suicide_texts.split()).most_common(10)
    words, counts = zip(*non_suicide_words)
    plt.bar(words, counts)
    plt.title('Top 10 Words - Non-Suicide Class')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Print basic statistics
    print("Dataset Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Classes: {df[label_col].value_counts().to_dict()}")
    print(f"Average text length: {df['text_length'].mean():.2f} characters")
    print(f"Average word count: {df['word_count'].mean():.2f} words")

perform_eda(df, LABEL_COL)


# Step 3: Train-Test Split
# Split the data
X_train, X_temp, y_train, y_temp = train_test_split(
    df["text_clean"], df[LABEL_COL], test_size=0.30, random_state=42, stratify=df[LABEL_COL]
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Training set: {len(X_train)} samples")
print(f"Validation set: {len(X_valid)} samples")
print(f"Test set: {len(X_test)} samples")


# Step 4: Basic Word Embedding Approaches
# 4.1 Bag of Words (BoW)
bow_vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, max_features=5000)
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_valid_bow = bow_vectorizer.transform(X_valid)
X_test_bow = bow_vectorizer.transform(X_test)

print("Bag of Words feature matrix shape:", X_train_bow.shape)

# 4.2 TF-IDF
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_valid_tfidf = tfidf_vectorizer.transform(X_valid)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print("TF-IDF feature matrix shape:", X_train_tfidf.shape)


# Step 5: Multiple Classification Models
def train_and_evaluate_models(X_train_vec, X_valid_vec, X_test_vec, y_train, y_valid, y_test, embedding_name):
    """Train multiple classifiers and evaluate their performance"""
    
    classifiers = {
        'LogisticRegression': LogisticRegression(max_iter=2000, random_state=42),
        'LinearSVC': LinearSVC(random_state=42)
    }
    
    results = []
    
    for clf_name, clf in classifiers.items():
        print(f"\n=== {embedding_name} + {clf_name} ===")
        
        # Train model
        start_time = time.time()
        clf.fit(X_train_vec, y_train)
        training_time = time.time() - start_time
        
        # Predictions
        y_valid_pred = clf.predict(X_valid_vec)
        y_test_pred = clf.predict(X_test_vec)
        
        # Validation metrics
        valid_accuracy = accuracy_score(y_valid, y_valid_pred)
        valid_f1 = f1_score(y_valid, y_valid_pred, average='macro', zero_division=0)
        
        # Test metrics
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
        
        # Detailed classification report for test set
        print("Test Set Classification Report:")
        print(classification_report(y_test, y_test_pred, digits=3))
        
        print(f"Training Time: {training_time:.2f} seconds")
        print(f"Validation - Accuracy: {valid_accuracy:.4f}, F1-macro: {valid_f1:.4f}")
        print(f"Test - Accuracy: {test_accuracy:.4f}, F1-macro: {test_f1:.4f}")
        
        # Store results
        results.append({
            'Embedding': embedding_name,
            'Classifier': clf_name,
            'Validation_Accuracy': valid_accuracy,
            'Validation_F1': valid_f1,
            'Test_Accuracy': test_accuracy,
            'Test_F1': test_f1,
            'Training_Time': training_time
        })
        
        # Plot confusion matrix for test set
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(y_test, y_test_pred, labels=["suicide", "non-suicide"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                   xticklabels=["suicide", "non-suicide"],
                   yticklabels=["suicide", "non-suicide"])
        plt.title(f"{embedding_name} + {clf_name}\nTest Set Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()
    
    return pd.DataFrame(results)

# Train with Bag of Words
print("="*60)
print("BAG OF WORDS RESULTS")
print("="*60)
bow_results = train_and_evaluate_models(X_train_bow, X_valid_bow, X_test_bow, y_train, y_valid, y_test, "BoW")

# Train with TF-IDF
print("="*60)
print("TF-IDF RESULTS")
print("="*60)
tfidf_results = train_and_evaluate_models(X_train_tfidf, X_valid_tfidf, X_test_tfidf, y_train, y_valid, y_test, "TF-IDF")


# Step 6: Advanced Word Embedding - BERT
# Install required packages if not already installed
# ! pip install transformers torch

import torch
from transformers import AutoTokenizer, AutoModel

def get_bert_embeddings(texts, model_name="distilbert-base-uncased", batch_size=32, max_length=128):
    """Get BERT embeddings for texts"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=max_length, 
                return_tensors="pt"
            ).to(device)
            
            outputs = model(**inputs)
            # Use the [CLS] token embedding as sentence representation
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            embeddings.append(cls_embedding.cpu().numpy())
    
    return np.vstack(embeddings)

# Get BERT embeddings (use subset for faster prototyping)
sample_size = min(2000, len(X_train))  # Adjust based on your computational resources
X_train_bert = get_bert_embeddings(X_train.tolist()[:sample_size])
X_valid_bert = get_bert_embeddings(X_valid.tolist())
X_test_bert = get_bert_embeddings(X_test.tolist())

print(f"BERT embeddings shape - Train: {X_train_bert.shape}, Valid: {X_valid_bert.shape}, Test: {X_test_bert.shape}")

# Train classifiers on BERT embeddings (Only Logistic Regression and Linear SVM)
bert_results = train_and_evaluate_models(X_train_bert, X_valid_bert, X_test_bert, 
                                        y_train[:sample_size], y_valid, y_test, "BERT")


# Step 8: Comprehensive Results Comparison
def plot_comparison_results(all_results):
    """Plot comparison of all models"""
    
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Create comprehensive comparison plots
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Test F1 scores
    plt.subplot(1, 3, 1)
    sns.barplot(data=combined_results, x='Embedding', y='Test_F1', hue='Classifier', 
                palette='viridis', edgecolor='black')
    plt.title('Test F1-Macro Score\nby Embedding and Classifier', fontsize=12, fontweight='bold')
    plt.xlabel('Embedding Method', fontweight='bold')
    plt.ylabel('F1-Macro Score', fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 2: Test Accuracy
    plt.subplot(1, 3, 2)
    sns.barplot(data=combined_results, x='Embedding', y='Test_Accuracy', hue='Classifier',
                palette='plasma', edgecolor='black')
    plt.title('Test Accuracy\nby Embedding and Classifier', fontsize=12, fontweight='bold')
    plt.xlabel('Embedding Method', fontweight='bold')
    plt.ylabel('Accuracy', fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 3: Training Time
    plt.subplot(1, 3, 3)
    sns.barplot(data=combined_results, x='Embedding', y='Training_Time', hue='Classifier',
                palette='coolwarm', edgecolor='black')
    plt.title('Training Time\nby Embedding and Classifier', fontsize=12, fontweight='bold')
    plt.xlabel('Embedding Method', fontweight='bold')
    plt.ylabel('Training Time (seconds)', fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print best performing models
    best_f1 = combined_results.loc[combined_results['Test_F1'].idxmax()]
    best_accuracy = combined_results.loc[combined_results['Test_Accuracy'].idxmax()]
    
    print("\n" + "="*60)
    print("BEST PERFORMING MODELS")
    print("="*60)
    print(f"Best F1-Macro: {best_f1['Embedding']} + {best_f1['Classifier']} - F1: {best_f1['Test_F1']:.4f}")
    print(f"Best Accuracy: {best_accuracy['Embedding']} + {best_accuracy['Classifier']} - Accuracy: {best_accuracy['Test_Accuracy']:.4f}")
    
    # Performance summary by embedding method
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY BY EMBEDDING METHOD")
    print("="*60)
    
    for embedding in combined_results['Embedding'].unique():
        embed_results = combined_results[combined_results['Embedding'] == embedding]
        avg_f1 = embed_results['Test_F1'].mean()
        avg_accuracy = embed_results['Test_Accuracy'].mean()
        print(f"{embedding}: Avg F1 = {avg_f1:.4f}, Avg Accuracy = {avg_accuracy:.4f}")
    
    return combined_results

# Combine all results and plot comparison
all_results = [bow_results, tfidf_results, bert_results]
final_results = plot_comparison_results(all_results)

# Display final results table
print("\nFINAL RESULTS SUMMARY:")
print(final_results[['Embedding', 'Classifier', 'Test_Accuracy', 'Test_F1', 'Training_Time']].round(4))


# Step 10: Performance Heatmaps for Accuracy and Test Error 
# ================================================================
def create_accuracy_error_heatmaps(results_df, save_path="accuracy_error_heatmaps_red_error.png"):
    """
    Accuracy → Green scale
    Test Error → Pure RED scale only
    Saves the figure as PNG (ready to attach)
    """
    results_df = results_df.copy()
    results_df['Test_Error'] = 1 - results_df['Test_Accuracy']

    # Pivot tables
    accuracy_pivot = results_df.pivot(index='Classifier', 
                                      columns='Embedding', 
                                      values='Test_Accuracy')
    error_pivot = results_df.pivot(index='Classifier', 
                                   columns='Embedding', 
                                   values='Test_Error')

    # Ensure exact order like in your original plot
    accuracy_pivot = accuracy_pivot.reindex(index=['LogisticRegression', 'LinearSVC'],
                                            columns=['BERT', 'BoW', 'TF-IDF'])
    error_pivot = error_pivot.reindex(index=['LogisticRegression', 'LinearSVC'],
                                      columns=['BERT', 'BoW', 'TF-IDF'])

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # ==================== ACCURACY HEATMAP (Green) ====================
    sns.heatmap(accuracy_pivot,
                annot=True, fmt='.4f',
                cmap='YlGn',                    # Yellow → Green (same as before)
                ax=ax1,
                cbar_kws={'label': 'Accuracy Score'},
                vmin=0.89, vmax=0.94,
                linewidths=0.5, linecolor='white',
                annot_kws={'size': 12, 'weight': 'bold'})

    ax1.set_title('ACCURACY SCORES HEATMAP', 
                  fontweight='bold', fontsize=14, pad=20)
    ax1.set_xlabel('Embedding Method', fontweight='bold')
    ax1.set_ylabel('Classifier', fontweight='bold')

    # ==================== TEST ERROR HEATMAP (Pure RED only) ====================
    sns.heatmap(error_pivot,
                annot=True, fmt='.4f',
                cmap='Reds',                    # ← ONLY RED shades
                ax=ax2,
                cbar_kws={'label': 'Error Rate'},
                vmin=0.06, vmax=0.11,           # Matches your original range
                linewidths=0.5, linecolor='white',
                annot_kws={'size': 12, 'weight': 'bold'})

    ax2.set_title('TEST ERROR RATES HEATMAP', 
                  fontweight='bold', fontsize=14, pad=20)
    ax2.set_xlabel('Embedding Method', fontweight='bold')
    ax2.set_ylabel('')

    # Final layout & save
    plt.tight_layout(pad=3.0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Heatmaps saved as → {save_path}")
    plt.show()

    return accuracy_pivot, error_pivot


# =============================================
# Run it (this will generate the file you can attach)
# =============================================
accuracy_scores, error_scores = create_accuracy_error_heatmaps(
    final_results,
    save_path="accuracy_error_heatmaps_GREEN_vs_PURE_RED.png"
)

