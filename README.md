# PhishingDetector

**Training and evaluation code for [PhishLens](https://github.com/AnzouK/PhishLens),
a multi-agent phishing email detector with explanations.**

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Model](https://img.shields.io/badge/Hugging%20Face-DistilBERT-yellow)](https://huggingface.co/AnzouKiona/phishlens-distilbert)
[![Agents](https://img.shields.io/badge/Hugging%20Face-Random%20Forests-yellow)](https://huggingface.co/AnzouKiona/phishlens-agents)
[![Runtime](https://img.shields.io/badge/Runtime-PhishLens-blueviolet)](https://github.com/AnzouK/PhishLens)

*Smart Phishing Detection System Using NLP Techniques.* B.Sc. final-year
project, Department of Cybersecurity, Nile University of Nigeria,
2025/2026.

The system scores an email with three agents and fuses their votes:

- a **text agent**, DistilBERT fine-tuned on the email body;
- a **URL agent**, a Random Forest over 23 features of the links;
- a **metadata agent**, a Random Forest over 37 header features
  (SPF, DKIM, DMARC, Reply-To mismatch, routing, subject).

LIME then shows which words pushed the text agent's decision.

**This repository is the research side:** the Colab notebooks, the
dataset loaders and the feature engineering used to train the three
agents. To install and use the detector (Chrome extension for Gmail,
FastAPI backend, live demo), go to
**[PhishLens](https://github.com/AnzouK/PhishLens)**.

## Results

Held-out test sets (stratified 20% split for the text agent):

| Agent | Model | Accuracy | F1 | ROC-AUC |
| --- | --- | --- | --- | --- |
| Text | DistilBERT, 3 epochs | 97.34% | 0.9665 | n/a |
| URL | Random Forest, 23 features | 99.34% | 0.9911 | 0.9969 |
| Metadata | Random Forest, 37 features | 99.92% | 0.9994 | 1.0000 |

The metadata score is best read as an upper bound: phishing and
legitimate samples come from different corpora, so header artefacts
(dates, relays) make the two classes easier to separate than in a real
inbox. See [Limitations](#limitations).

## Reproduce the training

Open the notebooks in Google Colab on a free T4 GPU runtime, in this
order:

1. **`DistilBERT_Phishing_Text_Agent.ipynb`**: loads the MeAJOR corpus
   from Hugging Face, fine-tunes DistilBERT for 3 epochs, evaluates on a
   stratified 20% test split, runs LIME and saves the model.
2. **`URL_Agent.ipynb`** and **`Metadata_Agent.ipynb`**: build the
   multi-source `.eml` corpus (Nazario, phishing_pot, Enron ham, and the
   Cisco Umbrella top-1m for legitimate URLs), train the Random Forests
   and report per-class metrics.

To reproduce the out-of-distribution augmentation described in section
4.5.1 of the report, paste the cell from `augmentation_cell.py` into the
DistilBERT notebook just before the train/test split and re-run from
there. It adds `AreLit/PhishNChips`,
`cybersectony/PhishingEmailDetectionv2.0` and the local
`synthetic_legit_emails.csv`, then deduplicates.

## Datasets

| Source | Role | Count |
| --- | --- | ---: |
| zefang-liu/phishing-email-dataset (MeAJOR corpus) | Text agent baseline | 18,650 |
| rf-peixoto/phishing_pot | Recent real-world phishing | varies |
| Nazario phishing corpus (2022 and later) | Modern phishing baseline | varies |
| SetFit/enron_spam (ham only) | Legitimate baseline | varies |
| AreLit/PhishNChips | Modern workplace legitimate mail | 1,333 |
| cybersectony/PhishingEmailDetectionv2.0 | Augmentation, legitimate | 11,322 |
| `synthetic_legit_emails.csv` (this repo) | Hand-written Nigerian-domain legitimate mail | 150 |
| Cisco Umbrella top-1m | Legitimate URLs for the URL agent | 10,000 |

After deduplication the augmented text corpus holds **29,555 emails**
(17,447 legitimate, 12,108 phishing).

## Trained models

The weights are too large for GitHub and live on Hugging Face:

- DistilBERT text agent (about 268 MB):
  [`AnzouKiona/phishlens-distilbert`](https://huggingface.co/AnzouKiona/phishlens-distilbert)
- URL and metadata Random Forests:
  [`AnzouKiona/phishlens-agents`](https://huggingface.co/AnzouKiona/phishlens-agents)

The Random Forests are published in the **skops** format
(`url_agent.skops`, `metadata_agent.skops`), which loads without pickle:
the loader only accepts scikit-learn, NumPy and SciPy types, so a
tampered file cannot run code. The original `.joblib` exports stay in
the repo for reproducibility; the notebooks still produce joblib, and
PhishLens ships a one-time converter
([`convert_agents_to_skops.py`](https://github.com/AnzouK/PhishLens/blob/main/backend/convert_agents_to_skops.py)).
The model card lives in [`model_cards/phishlens-agents.md`](model_cards/phishlens-agents.md).

The PhishLens backend downloads both at startup, so you only need them
here if you want to evaluate or retrain.

## Repository layout

```
DistilBERT_Phishing_Text_Agent.ipynb  text agent: training, evaluation, LIME
URL_Agent.ipynb                       URL agent: training and evaluation
Metadata_Agent.ipynb                  metadata agent: training and evaluation
dataset_handling.py                   loaders for every corpus above
email_preprocessing.py                .eml parsing, body and URL extraction
feature_extraction.py                 URL and header features (shared with PhishLens)
url_agent.py, metadata_agent.py       Random Forest wrappers (training, save, load)
augmentation_cell.py                  Colab cell for the augmentation step
synthetic_legit_emails.csv            150 hand-written legitimate emails
requirements.txt                      Python dependencies
model_cards/                          Hugging Face model cards
```

`feature_extraction.py` is copied into PhishLens and must stay identical
there: the Random Forests depend on its exact feature names and order.

## Limitations

1. **English only.** Tokenizer and corpora are English; French, Hausa
   and Yoruba phishing are out of distribution.
2. **Offline evaluation.** All numbers come from held-out and
   qualitative test sets, not from a live inbox.
3. **Corpus artefacts.** Phishing and legitimate samples come from
   different sources and years, which inflates the header-based scores.
4. **Niche false positives.** Promotional and specialised recruitment
   emails outside the augmentation data are still sometimes flagged.
5. **Attachments are not used in training.** The models only read the
   text parts of an email. PhishLens adds PDF and HTML attachment
   scanning at runtime (since v1.8), on top of these models.

The runtime mitigations (sender authentication, threat intelligence,
trust paths) and a full threat model are documented in the
[PhishLens docs](https://github.com/AnzouK/PhishLens/tree/main/docs).

## License and citation

Released under the [MIT License](LICENSE) with the agreement of the
project supervisor. If you reuse or reference this work, please credit
it as a B.Sc. final-year project of the Department of Cybersecurity,
Faculty of Computing, Nile University of Nigeria, 2025/2026.
