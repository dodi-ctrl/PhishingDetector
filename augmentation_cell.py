# =====================================================
# AUGMENT TRAINING SET WITH MODERN LEGITIMATE EMAILS
# Insert this cell BEFORE the stratified train/test split
# =====================================================
# Sources:
#   - AreLit/PhishNChips    : 1000 modern workplace legit emails (synthetic but realistic)
#   - cybersectony/PhishingEmailDetectionv2.0 : ~11k legit_email samples (label=0, mix of pro emails)
#   - synthetic_legit_emails.csv : our 150 hand-templated emails (banking, university, etc.)
#
# Target after augmentation: ~12k legit modern emails added on top of current corpus.
# Expected effect: drastically reduces false positives on real bank/university/workplace emails
# while preserving phishing detection (we don't touch the phishing class).

!pip install -q datasets

from datasets import load_dataset
import pandas as pd

print("=" * 60)
print("STEP 1/4 : Loading PhishNChips (modern workplace legit emails)")
print("=" * 60)

import json

phishnchips = load_dataset("AreLit/PhishNChips", "emails")
print("PhishNChips splits:", list(phishnchips.keys()))

# Combine all splits
phishnchips_df_list = [phishnchips[s].to_pandas() for s in phishnchips.keys()]
phishnchips_full = pd.concat(phishnchips_df_list, ignore_index=True)
print(f"PhishNChips total rows: {len(phishnchips_full)}")
print(f"PhishNChips columns: {list(phishnchips_full.columns)}")
print(f"phish_label distribution:\n{phishnchips_full['phish_label'].value_counts()}")

# Filter to legitimate emails only (phish_label == 0)
phishnchips_legit_raw = phishnchips_full[phishnchips_full['phish_label'] == 0].copy()
print(f"PhishNChips legit-only rows (raw): {len(phishnchips_legit_raw)}")

# email_content is a JSON string with fields like sender, subject, body...
# We extract a flat text representation: "Subject: ...\nFrom: ...\n\n<body>"
def parse_email_json(raw):
    if pd.isna(raw):
        return ""
    if isinstance(raw, str):
        # Try parsing as JSON; if it fails, return raw as-is (already flat text).
        try:
            obj = json.loads(raw)
        except Exception:
            return raw
    elif isinstance(raw, dict):
        obj = raw
    else:
        return str(raw)

    # Walk common keys to assemble a readable email
    parts = []
    for key in ['subject', 'Subject', 'subj']:
        if key in obj and obj[key]:
            parts.append(f"Subject: {obj[key]}")
            break
    for key in ['sender', 'from', 'From']:
        if key in obj and obj[key]:
            parts.append(f"From: {obj[key]}")
            break
    for key in ['recipient', 'to', 'To']:
        if key in obj and obj[key]:
            parts.append(f"To: {obj[key]}")
            break
    body_keys = ['body', 'content', 'text', 'message', 'email_body', 'plain_text', 'html']
    body = ""
    for key in body_keys:
        if key in obj and obj[key]:
            body = str(obj[key])
            break
    if not body:
        # Fallback: dump whatever else is there
        body = " ".join(f"{k}: {v}" for k, v in obj.items() if k not in ['subject','Subject','sender','from','From','recipient','to','To'])
    parts.append("")
    parts.append(body)
    return "\n".join(parts)

phishnchips_legit_raw['text'] = phishnchips_legit_raw['email_content'].apply(parse_email_json)
phishnchips_legit = phishnchips_legit_raw[['text']].copy()
phishnchips_legit['label'] = 0
print(f"PhishNChips legit-only rows (parsed): {len(phishnchips_legit)}")
print(f"Sample parsed legit email:\n{phishnchips_legit['text'].iloc[0][:300]}\n...")


print("=" * 60)
print("STEP 2/4 : Loading cybersectony PhishingEmailDetectionv2.0")
print("=" * 60)

cyber = load_dataset("cybersectony/PhishingEmailDetectionv2.0")
print("Cybersectony splits:", list(cyber.keys()))
print("Cybersectony columns:", cyber['train'].column_names)

# Build a unified DataFrame from all splits
cyber_df_list = []
for split_name in cyber.keys():
    cyber_df_list.append(cyber[split_name].to_pandas())
cyber_full = pd.concat(cyber_df_list, ignore_index=True)
print(f"Cybersectony total rows: {len(cyber_full)}")
print(f"Label distribution:\n{cyber_full['label'].value_counts()}")

# Schema:
#   0 = legitimate_email  <-- WE WANT THIS
#   1 = phishing_email
#   2 = legitimate_url
#   3 = phishing_url
cyber_legit_email = cyber_full[cyber_full['label'] == 0][['content']].copy()
cyber_legit_email.columns = ['text']
cyber_legit_email['label'] = 0
print(f"Cybersectony legit_email rows: {len(cyber_legit_email)}")


print("=" * 60)
print("STEP 3/4 : Loading our 150 hand-templated legit emails")
print("=" * 60)

# Upload synthetic_legit_emails.csv if not already in the env
import os
if not os.path.exists('synthetic_legit_emails.csv'):
    print("synthetic_legit_emails.csv not found — please upload it.")
    from google.colab import files
    uploaded = files.upload()
    # the uploader puts files in cwd

synth_df = pd.read_csv('synthetic_legit_emails.csv')
synth_df = synth_df[['text', 'label']]
synth_df['label'] = synth_df['label'].astype(int)
print(f"Hand-templated synthetic rows: {len(synth_df)}")


print("=" * 60)
print("STEP 4/4 : Merging into the existing training corpus 'df'")
print("=" * 60)

# Drop empty / very short emails (less than 10 chars) to avoid garbage
def clean(d):
    d = d.dropna(subset=['text']).copy()
    d['text'] = d['text'].astype(str)
    d = d[d['text'].str.len() > 10].reset_index(drop=True)
    return d

phishnchips_legit = clean(phishnchips_legit)
cyber_legit_email = clean(cyber_legit_email)
synth_df = clean(synth_df)

print(f"Original df size:           {len(df)}")
print(f"PhishNChips legit to add:   {len(phishnchips_legit)}")
print(f"Cybersectony legit to add:  {len(cyber_legit_email)}")
print(f"Hand-templated to add:      {len(synth_df)}")

augmented = pd.concat([
    df,
    phishnchips_legit,
    cyber_legit_email,
    synth_df,
], ignore_index=True)

# Drop exact duplicates (some Enron emails may already be in df)
before = len(augmented)
augmented = augmented.drop_duplicates(subset=['text']).reset_index(drop=True)
after = len(augmented)
print(f"After concat:               {before}")
print(f"After dedup:                {after}  (removed {before-after} duplicates)")

# Shuffle
augmented = augmented.sample(frac=1, random_state=42).reset_index(drop=True)

# Replace df
df = augmented

print()
print("=" * 60)
print("FINAL CLASS DISTRIBUTION")
print("=" * 60)
print(df['label'].value_counts())
print()
print("Class balance ratio (legit/phishing):", round(
    df[df['label']==0].shape[0] / max(df[df['label']==1].shape[0], 1), 3
))
print()
print("✅ Augmentation complete. Continue to the train/test split cell.")
