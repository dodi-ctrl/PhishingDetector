"""
Phishing Detection System - Accuracy Test
Using MeAJOR Corpus (zefang-liu/phishing-email-dataset)
Configuration: 20,000 total samples | 70% Legitimate / 30% Phishing

Author: Final Year Project
Dataset: MeAJOR Corpus - Peer-reviewed at LREC-COLING 2024
"""

import os
import re
import mailbox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from email import policy as email_policy
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Import your existing feature extractor
from feature_extraction import FeatureExtractor

# ============================================
# STEP 1: LOAD MeAJOR CORPUS FROM HUGGINGFACE
# ============================================

def load_meajor_dataset(total_samples=20000, legit_ratio=0.7):
    """
    Load MeAJOR Corpus with specific class distribution.
    
    Parameters:
    -----------
    total_samples : int
        Total number of samples to use (default: 20000)
    legit_ratio : float
        Ratio of legitimate emails (0.7 = 70% legit, 30% phishing)
    
    Returns:
    --------
    DataFrame with selected samples
    """
    print("="*70)
    print("LOADING MeAJOR CORPUS FROM HUGGINGFACE")
    print("="*70)
    
    try:
        # Load the dataset from HuggingFace
        print("\nDownloading MeAJOR Corpus...")
        dataset = load_dataset("zefang-liu/phishing-email-dataset")
        
        # Check available splits
        if 'train' in dataset:
            df = pd.DataFrame(dataset['train'])
        else:
            split_name = list(dataset.keys())[0]
            df = pd.DataFrame(dataset[split_name])
        
        print(f"✓ Total available samples: {len(df)}")
        
        # Identify label column (check common names)
        label_col = None
        possible_labels = ['label', 'is_phishing', 'target', 'class', 'phishing', 'Email Type']
        for col in possible_labels:
            if col in df.columns:
                label_col = col
                break

        if label_col is None:
            print(f"Available columns: {list(df.columns)}")
            raise ValueError("Could not find label column")

        print(f"✓ Using label column: '{label_col}'")

        # Convert string labels to numeric if needed (e.g. "Safe Email" / "Phishing Email")
        # Use is_integer_dtype rather than == object to handle PyArrow-backed string columns
        if not pd.api.types.is_integer_dtype(df[label_col]):
            unique_vals = df[label_col].unique()
            print(f"  String labels detected: {list(unique_vals)}")
            phishing_str = next(
                (v for v in unique_vals if 'phish' in str(v).lower()), None
            )
            if phishing_str is None:
                raise ValueError(f"Cannot identify phishing label from values: {list(unique_vals)}")
            df['label'] = (df[label_col] == phishing_str).astype(int)
            label_col = 'label'
            print(f"  Converted: '{phishing_str}' → 1, others → 0")

        # Check current distribution
        legit_count = (df[label_col] == 0).sum()
        phishing_count = (df[label_col] == 1).sum()
        total_available = len(df)
        
        print(f"\nOriginal dataset distribution:")
        print(f"  Legitimate (0): {legit_count} ({legit_count/total_available*100:.1f}%)")
        print(f"  Phishing (1):   {phishing_count} ({phishing_count/total_available*100:.1f}%)")
        
        # Calculate how many of each class we need
        target_legit = int(total_samples * legit_ratio)
        target_phishing = total_samples - target_legit
        
        print(f"\nTarget distribution for testing:")
        print(f"  Legitimate (0): {target_legit} ({legit_ratio*100:.0f}%)")
        print(f"  Phishing (1):   {target_phishing} ({(1-legit_ratio)*100:.0f}%)")
        print(f"  Total: {total_samples} samples")
        
        # Sample legitimate emails
        legit_df = df[df[label_col] == 0]
        if len(legit_df) < target_legit:
            print(f"⚠ Warning: Only {len(legit_df)} legitimate samples available. Using all.")
            sampled_legit = legit_df
        else:
            sampled_legit = legit_df.sample(n=target_legit, random_state=42)
        
        # Sample phishing emails
        phishing_df = df[df[label_col] == 1]
        if len(phishing_df) < target_phishing:
            print(f"⚠ Warning: Only {len(phishing_df)} phishing samples available. Using all.")
            sampled_phishing = phishing_df
        else:
            sampled_phishing = phishing_df.sample(n=target_phishing, random_state=42)
        
        # Combine and shuffle
        final_df = pd.concat([sampled_legit, sampled_phishing], ignore_index=True)
        final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Verify final distribution
        final_legit = (final_df[label_col] == 0).sum()
        final_phishing = (final_df[label_col] == 1).sum()
        
        print(f"\n✓ Final dataset created:")
        print(f"  Legitimate: {final_legit} ({final_legit/len(final_df)*100:.1f}%)")
        print(f"  Phishing:   {final_phishing} ({final_phishing/len(final_df)*100:.1f}%)")
        print(f"  Total: {len(final_df)} samples")
        
        return final_df, label_col
        
    except Exception as e:
        print(f"\n✗ Error loading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you have internet connection")
        print("2. Run: pip install datasets")
        print("3. The dataset may require acceptance of terms on HuggingFace")
        return None, None

# ============================================
# STEP 2: EXTRACT FEATURES USING YOUR EXTRACTOR
# ============================================

def extract_features_from_emails(df, text_column, extractor):
    """
    Extract features from email texts using FeatureExtractor.
    """
    all_features = []
    
    print("\n" + "="*70)
    print("EXTRACTING FEATURES FROM EMAILS")
    print("="*70)
    
    texts = df[text_column].tolist()
    
    for i, text in enumerate(texts):
        if i % 2000 == 0 and i > 0:
            print(f"  Processed {i}/{len(texts)} emails...")
        
        # Handle NaN or None values
        if pd.isna(text) or text is None:
            text = ""
        
        # Extract features using your existing FeatureExtractor
        try:
            text_features = extractor.extract_text_features(str(text))
            url_features = extractor.extract_url_features(str(text))
            
            # Combine features
            combined = {**text_features, **url_features}
            all_features.append(combined)
        except Exception as e:
            # If feature extraction fails, add empty features
            print(f"  Warning: Failed to extract features for sample {i}: {e}")
            all_features.append({})
    
    features_df = pd.DataFrame(all_features)
    
    # Fill any NaN values with 0
    features_df = features_df.fillna(0)
    
    print(f"\n✓ Extracted {len(features_df.columns)} features from {len(features_df)} emails")
    
    return features_df

# ============================================
# STEP 3: TRAIN AND EVALUATE MODEL
# ============================================

def train_and_evaluate(features_df, labels, test_ratio=0.3):
    """
    Train Random Forest model and evaluate accuracy.
    """
    print("\n" + "="*70)
    print("MODEL TRAINING & EVALUATION")
    print("="*70)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, labels, test_size=test_ratio,
        random_state=42, stratify=labels
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"  - Legitimate in test: {(y_test == 0).sum()}")
    print(f"  - Phishing in test: {(y_test == 1).sum()}")
    
    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    print("\nTraining Random Forest classifier...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Print results
    print("\n" + "-"*70)
    print("ACCURACY TEST RESULTS")
    print("-"*70)
    print(f"\n{'ACCURACY:':25} {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'PRECISION:':25} {precision:.4f} ({precision*100:.2f}%)")
    print(f"{'RECALL:':25} {recall:.4f} ({recall*100:.2f}%)")
    print(f"{'F1-SCORE:':25} {f1:.4f} ({f1*100:.2f}%)")
    print(f"{'ROC-AUC:':25} {roc_auc:.4f}")
    
    print(f"\n{'False Positive Rate:':25} {fp/(fp+tn):.4f}")
    print(f"{'False Negative Rate:':25} {fn/(fn+tp):.4f}")
    
    print("\n" + "-"*70)
    print("CONFUSION MATRIX")
    print("-"*70)
    print(f"{'':20} {'Predicted Legit':>18} {'Predicted Phishing':>20}")
    print(f"{'Actual Legit':20} {tn:>18} {fp:>20}")
    print(f"{'Actual Phishing':20} {fn:>18} {tp:>20}")
    
    print("\n" + "-"*70)
    print("CLASSIFICATION REPORT")
    print("-"*70)
    print(classification_report(y_test, y_pred, 
                               target_names=['Legitimate', 'Phishing'],
                               digits=4))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features_df.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n" + "-"*70)
    print("TOP 15 MOST IMPORTANT FEATURES")
    print("-"*70)
    for i, row in feature_importance.head(15).iterrows():
        print(f"{row['feature']:35} {row['importance']:.6f}")
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'y_test': y_test,
        'y_pred_proba': y_pred_proba,
        'feature_importance': feature_importance
    }
    
    return model, results

# ============================================
# STEP 4: VISUALIZE RESULTS
# ============================================

def plot_results(results, total_samples=20000, legit_ratio=0.7):
    """
    Generate visualization plots for the accuracy test.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Phishing Detection System - MeAJOR Corpus\n'
                 f'{total_samples} samples ({legit_ratio*100:.0f}% Legitimate / {(1-legit_ratio)*100:.0f}% Phishing)', 
                 fontsize=12, fontweight='bold')
    
    # 1. Confusion Matrix Heatmap
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'],
                ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_ylabel('Actual')
    axes[0, 0].set_xlabel('Predicted')
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(results['y_test'], results['y_pred_proba'])
    axes[0, 1].plot(fpr, tpr, 'b-', linewidth=2, 
                    label=f'ROC (AUC = {results["roc_auc"]:.4f})')
    axes[0, 1].plot([0, 1], [0, 1], 'r--', label='Random Classifier')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Performance Metrics Bar Chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [results['accuracy'], results['precision'], 
              results['recall'], results['f1_score']]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    bars = axes[1, 0].bar(metrics, values, color=colors)
    axes[1, 0].set_ylim([0, 1.1])
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Performance Metrics')
    axes[1, 0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, values):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                       f'{val:.4f}', ha='center', fontweight='bold')
    
    # 4. Feature Importance (Top 10)
    feature_imp = results['feature_importance'].head(10)
    axes[1, 1].barh(range(len(feature_imp)), feature_imp['importance'].values)
    axes[1, 1].set_yticks(range(len(feature_imp)))
    axes[1, 1].set_yticklabels(feature_imp['feature'].values, fontsize=9)
    axes[1, 1].set_xlabel('Importance')
    axes[1, 1].set_title('Top 10 Most Important Features')
    axes[1, 1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('meajor_accuracy_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Results plot saved as 'meajor_accuracy_results.png'")
    plt.show()

# ============================================
# MAIN EXECUTION
# ============================================

def main():
    # Configuration
    TOTAL_SAMPLES = 20000
    LEGIT_RATIO = 0.7  # 70% legitimate, 30% phishing
    
    print("="*70)
    print("PHISHING DETECTION SYSTEM - ACCURACY TEST")
    print("Dataset: MeAJOR Corpus (zefang-liu/phishing-email-dataset)")
    print(f"Configuration: {TOTAL_SAMPLES} total samples")
    print(f"                {LEGIT_RATIO*100:.0f}% Legitimate / {(1-LEGIT_RATIO)*100:.0f}% Phishing")
    print("="*70)
    
    # Step 1: Load dataset
    df, label_col = load_meajor_dataset(
        total_samples=TOTAL_SAMPLES,
        legit_ratio=LEGIT_RATIO
    )
    
    if df is None:
        return
    
    # Identify text column (email body)
    text_column = None
    possible_text_cols = ['text', 'body', 'email_body', 'content', 'email', 'Email Text']
    for col in possible_text_cols:
        if col in df.columns:
            text_column = col
            break
    
    if text_column is None:
        print(f"Available columns: {list(df.columns)}")
        print("Please identify which column contains the email text.")
        return
    
    print(f"\n✓ Using text column: '{text_column}'")
    
    # Step 2: Extract features
    extractor = FeatureExtractor()
    features_df = extract_features_from_emails(df, text_column, extractor)
    labels = df[label_col].values
    
    # Step 3: Train and evaluate
    model, results = train_and_evaluate(features_df, labels, test_ratio=0.3)
    
    # Step 4: Plot results
    plot_results(results, TOTAL_SAMPLES, LEGIT_RATIO)
    
    # Step 5: Save results to JSON
    results_summary = {
        'dataset': 'MeAJOR Corpus (zefang-liu/phishing-email-dataset)',
        'citation': 'Mendes et al., LREC-COLING 2024',
        'total_samples': TOTAL_SAMPLES,
        'legitimate_ratio': LEGIT_RATIO,
        'phishing_ratio': 1 - LEGIT_RATIO,
        'test_samples': int(TOTAL_SAMPLES * 0.3),
        'accuracy': float(results['accuracy']),
        'precision': float(results['precision']),
        'recall': float(results['recall']),
        'f1_score': float(results['f1_score']),
        'roc_auc': float(results['roc_auc'])
    }
    
    with open('meajor_accuracy_results.json', 'w') as f:
        json.dump(results_summary, f, indent=4)
    print("\n✓ Results saved to 'meajor_accuracy_results.json'")
    
    print("\n" + "="*70)
    print("ACCURACY TEST COMPLETE")
    print("="*70)
    print("\nSummary for your report:")
    print(f"  Dataset: MeAJOR Corpus (Peer-reviewed, LREC-COLING 2024)")
    print(f"  Total samples: {TOTAL_SAMPLES}")
    print(f"  Distribution: {LEGIT_RATIO*100:.0f}% Legitimate, {(1-LEGIT_RATIO)*100:.0f}% Phishing")
    print(f"  Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"  F1-Score: {results['f1_score']:.4f}")

if __name__ == "__main__":
    main()


# ============================================================
# MULTI-CORPUS LOADERS

# ============================================================

def load_eml_directory(directory, label):
    """
    Walk a directory and load all .eml files as raw bytes.
    Returns list of (raw_bytes, label) tuples.
    """
    records = []
    if not os.path.isdir(directory):
        print(f"  Warning: directory not found: {directory}")
        return records

    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.lower().endswith('.eml'):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, 'rb') as f:
                        records.append((f.read(), label))
                except Exception as e:
                    print(f"  Warning: could not read {fpath}: {e}")

    print(f"  Loaded {len(records)} .eml files from {directory} (label={label})")
    return records


def load_mbox(mbox_path, label, after_year=None):
    """
    Parse an mbox file and return list of (raw_bytes, label) tuples.
    Optionally filter to messages from after_year onwards.
    """
    records = []
    if not os.path.isfile(mbox_path):
        print(f"  Warning: mbox not found: {mbox_path}")
        return records

    mbox = mailbox.mbox(mbox_path)
    for msg in mbox:
        try:
            if after_year is not None:
                try:
                    dt = parsedate_to_datetime(msg.get('Date', ''))
                    if dt.year < after_year:
                        continue
                except Exception:
                    pass  # include if date can't be parsed
            records.append((msg.as_bytes(), label))
        except Exception:
            pass

    print(f"  Loaded {len(records)} messages from {mbox_path} (label={label})")
    return records


def load_enron_ham_from_huggingface(max_samples=5000):
    """
    Load Enron ham from SetFit/enron_spam on HuggingFace.
    The HuggingFace version contains text body + subject, not full RFC 2822
    headers, so we reconstruct a minimal envelope for the parser.
    Returns list of (raw_bytes, label=0) tuples.
    """
    print("  Loading Enron ham from HuggingFace (SetFit/enron_spam)...")
    try:
        ds = load_dataset("SetFit/enron_spam", split="train")
    except Exception as e:
        print(f"  Warning: could not load Enron dataset: {e}")
        return []

    records = []
    for row in ds:
        if row.get('label', 1) != 0:   # 0 = ham
            continue

        subject = row.get('subject', '') or ''
        text    = row.get('text', '')    or ''
        sender  = row.get('sender', 'enron@enron.com') or 'enron@enron.com'

        # Minimal RFC 2822 envelope — no auth headers, no Received chain.
        # Metadata features that depend on SPF/DKIM/DMARC will be zero,
        # which is the correct baseline for this corpus.
        raw = (
            f"From: {sender}\r\n"
            f"Subject: {subject}\r\n"
            f"Date: Mon, 01 Jan 2001 12:00:00 +0000\r\n"
            f"Message-ID: <enron-ham@enron.com>\r\n"
            f"\r\n"
            f"{text}"
        ).encode('utf-8', errors='replace')

        records.append((raw, 0))
        if len(records) >= max_samples:
            break

    print(f"  Loaded {len(records)} Enron ham emails (label=0)")
    return records


def load_phishtank_urls(csv_path, max_urls=50000):
    """
    Load PhishTank CSV (column: 'url'). Returns list of (url, label=1) tuples.
    """
    if not os.path.isfile(csv_path):
        print(f"  Warning: PhishTank CSV not found: {csv_path}")
        return []
    try:
        df = pd.read_csv(csv_path, usecols=['url'], nrows=max_urls)
        records = [(u, 1) for u in df['url'].dropna().tolist()]
        print(f"  Loaded {len(records)} phishing URLs from PhishTank")
        return records
    except Exception as e:
        print(f"  Warning: could not load PhishTank CSV: {e}")
        return []


def load_umbrella_domains(csv_path, limit=50000):
    """
    Load Cisco Umbrella top-1M CSV (rank,domain). Returns list of
    ('https://<domain>/', label=0) tuples — minimal URL for feature extraction.
    """
    if not os.path.isfile(csv_path):
        print(f"  Warning: Umbrella CSV not found: {csv_path}")
        return []
    try:
        df = pd.read_csv(csv_path, header=None, names=['rank', 'domain'], nrows=limit)
        records = [(f"https://{row['domain']}/", 0)
                   for _, row in df.iterrows() if pd.notna(row['domain'])]
        print(f"  Loaded {len(records)} legitimate domains from Umbrella top-1M")
        return records
    except Exception as e:
        print(f"  Warning: could not load Umbrella CSV: {e}")
        return []


# ============================================================
# CORPUS BUILDERS
# ============================================================

def build_eml_corpus(
    phishing_dirs=None,
    phishing_mbox_paths=None,
    enron_max=5000,
    nazario_after_year=2022,
):
    """
    Assemble a unified .eml corpus for the metadata and URL agents.

    Parameters
    ----------
    phishing_dirs : list[str]
        Directories containing phishing .eml files (e.g. phishing_pot/emails/).
    phishing_mbox_paths : list[str]
        Paths to phishing mbox files (e.g. phishing3.mbox from Nazario).
    enron_max : int
        Cap on Enron ham emails loaded from HuggingFace.
    nazario_after_year : int
        Discard Nazario emails older than this year.

    Returns
    -------
    pd.DataFrame  columns: raw_bytes, label, source
    """
    print("\n" + "="*70)
    print("BUILDING EML CORPUS (phishing_pot + Nazario + Enron)")
    print("="*70)

    all_records = []

    if phishing_dirs:
        for d in phishing_dirs:
            for raw, lbl in load_eml_directory(d, label=1):
                all_records.append({'raw_bytes': raw, 'label': lbl, 'source': 'phishing_pot'})

    if phishing_mbox_paths:
        for path in phishing_mbox_paths:
            for raw, lbl in load_mbox(path, label=1, after_year=nazario_after_year):
                all_records.append({'raw_bytes': raw, 'label': lbl, 'source': 'nazario'})

    for raw, lbl in load_enron_ham_from_huggingface(max_samples=enron_max):
        all_records.append({'raw_bytes': raw, 'label': lbl, 'source': 'enron'})

    if not all_records:
        print("\nWarning: No emails loaded — check your data paths.")
        return pd.DataFrame(columns=['raw_bytes', 'label', 'source'])

    df = pd.DataFrame(all_records)
    phishing_n = (df['label'] == 1).sum()
    legit_n    = (df['label'] == 0).sum()

    print(f"\n✓ EML corpus built:")
    print(f"  Phishing:   {phishing_n}")
    print(f"  Legitimate: {legit_n}")
    print(f"  Total:      {len(df)}")
    print(f"  Sources:    {df['source'].value_counts().to_dict()}")
    return df


def build_url_corpus(
    eml_df=None,
    phishtank_csv=None,
    umbrella_csv=None,
    umbrella_limit=50000,
):
    """
    Build a URL corpus combining:
      - URLs extracted from the EML corpus (phishing + ham emails)
      - PhishTank verified phishing URLs
      - Cisco Umbrella legitimate domains

    Parameters
    ----------
    eml_df : pd.DataFrame or None
        Output of build_eml_corpus(). Pass None to skip EML URLs.
    phishtank_csv : str or None
        Path to PhishTank online-valid.csv.
    umbrella_csv : str or None
        Path to Umbrella top-1m.csv.
    umbrella_limit : int
        Maximum Umbrella rows to load.

    Returns
    -------
    pd.DataFrame  columns: url_text, label, source
    """
    print("\n" + "="*70)
    print("BUILDING URL CORPUS (EML + PhishTank + Umbrella)")
    print("="*70)

    all_records = []

    if eml_df is not None and len(eml_df) > 0:
        print("  Extracting URLs from EML corpus...")
        url_count_before = 0
        for _, row in eml_df.iterrows():
            try:
                body = _extract_text_from_raw_email(row['raw_bytes'])
                urls = re.findall(r'https?://[^\s<>"\']+', body)
                for url in urls:
                    all_records.append({
                        'url_text': url,
                        'label': row['label'],
                        'source': f"eml_{row.get('source', 'unknown')}",
                    })
            except Exception:
                pass
        print(f"  Extracted {len(all_records) - url_count_before} URLs from EML corpus")

    if phishtank_csv:
        for url, lbl in load_phishtank_urls(phishtank_csv):
            all_records.append({'url_text': url, 'label': lbl, 'source': 'phishtank'})

    if umbrella_csv:
        for url, lbl in load_umbrella_domains(umbrella_csv, limit=umbrella_limit):
            all_records.append({'url_text': url, 'label': lbl, 'source': 'umbrella'})

    if not all_records:
        print("\nWarning: No URLs loaded — check your data paths.")
        return pd.DataFrame(columns=['url_text', 'label', 'source'])

    df = pd.DataFrame(all_records)
    phishing_n = (df['label'] == 1).sum()
    legit_n    = (df['label'] == 0).sum()

    print(f"\n✓ URL corpus built:")
    print(f"  Phishing:   {phishing_n}")
    print(f"  Legitimate: {legit_n}")
    print(f"  Total:      {len(df)}")
    print(f"  Sources:    {df['source'].value_counts().to_dict()}")
    return df


def extract_body_text_from_eml_corpus(eml_df):
    """
    Extract plain-text body from each .eml in eml_df for DistilBERT fine-tuning.
    Returns pd.DataFrame with columns: text, label, source — suitable for
    concatenation with the MeAJOR CSV corpus.
    """
    rows = []
    for _, row in eml_df.iterrows():
        try:
            text = _extract_text_from_raw_email(row['raw_bytes'])
        except Exception:
            text = ''
        if text.strip():
            rows.append({'text': text, 'label': row['label'], 'source': row.get('source', 'eml')})
    return pd.DataFrame(rows)


# ============================================================
# FEATURE EXTRACTION FROM CORPORA
# ============================================================

def extract_metadata_features_from_eml_corpus(eml_df, extractor):
    """
    Run extract_metadata_features() on every raw email in eml_df.

    Parameters
    ----------
    eml_df : pd.DataFrame   columns: raw_bytes, label
    extractor : FeatureExtractor

    Returns
    -------
    tuple: (features_df, labels_array)
    """
    print("\n" + "="*70)
    print("EXTRACTING METADATA FEATURES FROM EML CORPUS")
    print("="*70)

    all_features = []
    total = len(eml_df)

    for i, (_, row) in enumerate(eml_df.iterrows()):
        if i % 500 == 0 and i > 0:
            print(f"  Processed {i}/{total} emails...")
        try:
            raw = row['raw_bytes']
            if isinstance(raw, str):
                raw = raw.encode('utf-8', errors='replace')
            all_features.append(extractor.extract_metadata_features(raw))
        except Exception:
            all_features.append(extractor._get_default_metadata_features())

    features_df = pd.DataFrame(all_features).fillna(0)
    labels = eml_df['label'].values

    print(f"\n✓ Extracted {len(features_df.columns)} metadata features from {len(features_df)} emails")
    return features_df, labels


def extract_url_features_from_url_corpus(url_df, extractor):
    """
    Run extract_url_features() on every entry in url_df.
    Each row's url_text is treated as the 'email body' — the regex inside
    extract_url_features finds the URL and processes it.

    Parameters
    ----------
    url_df : pd.DataFrame   columns: url_text, label
    extractor : FeatureExtractor

    Returns
    -------
    tuple: (features_df, labels_array)
    """
    print("\n" + "="*70)
    print("EXTRACTING URL FEATURES FROM URL CORPUS")
    print("="*70)

    all_features = []
    total = len(url_df)

    for i, (_, row) in enumerate(url_df.iterrows()):
        if i % 5000 == 0 and i > 0:
            print(f"  Processed {i}/{total} URLs...")
        try:
            all_features.append(extractor.extract_url_features(str(row['url_text'])))
        except Exception:
            all_features.append(extractor._get_default_url_features())

    features_df = pd.DataFrame(all_features).fillna(0)
    labels = url_df['label'].values

    print(f"\n✓ Extracted {len(features_df.columns)} URL features from {len(features_df)} entries")
    return features_df, labels


# ============================================================
# INTERNAL HELPER
# ============================================================

def _extract_text_from_raw_email(raw):
    """Extract plain-text body from raw RFC 2822 bytes or string."""
    try:
        if isinstance(raw, str):
            raw = raw.encode('utf-8', errors='replace')
        msg = BytesParser(policy=email_policy.default).parsebytes(raw)
        plain, html = None, None

        if msg.is_multipart():
            for part in msg.walk():
                ct = part.get_content_type()
                if ct == 'text/plain' and plain is None:
                    plain = part.get_content()
                elif ct == 'text/html' and html is None:
                    html = part.get_content()
        else:
            ct = msg.get_content_type()
            if ct == 'text/plain':
                plain = msg.get_content()
            elif ct == 'text/html':
                html = msg.get_content()

        if plain:
            return plain
        if html:
            return re.sub(r'<[^>]+>', ' ', html)
    except Exception:
        pass
    # Last-resort: decode raw bytes as text
    if isinstance(raw, bytes):
        return raw.decode('utf-8', errors='replace')
    return str(raw)
