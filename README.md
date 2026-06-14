# Smart Phishing Detection System Using NLP Techniques

**B.Sc. Final Year Project**
Department of Cybersecurity, Faculty of Computing
Nile University of Nigeria — 2025/2026 Academic Session

A multi-agent phishing detection system combining a fine-tuned DistilBERT
text classifier, two Random Forest agents over URL and metadata features,
a weighted decision-fusion layer, and a LIME explainability module that
surfaces a human-readable rationale for every prediction.

> 🛡 **Looking for the Chrome extension or the deployable backend?**
> Those live in a separate, lighter repository — see
> **[PhishLens](https://github.com/Sonje03/PhishLens)**.
> This repo holds the **training notebooks and per-agent code** only;
> PhishLens is what you clone if you just want to install and run the system.

---

## At a glance

| Agent           | Model              | Accuracy | F1     | ROC-AUC |
|-----------------|--------------------|----------|--------|---------|
| Text Agent      | DistilBERT         | 97.34%   | 0.9665 | —       |
| URL Agent       | Random Forest      | 99.34%   | 0.9911 | 0.9969  |
| Metadata Agent  | Random Forest      | 99.92%   | 0.9994 | 1.0000  |

---

## Repository layout

```
PhishingDetector/
├── README.md                                    (this file)
├── requirements.txt                             Python dependencies
├── .gitignore
│
├── DistilBERT_Phishing_Text_Agent.ipynb         Text agent training + LIME
├── Metadata_Agent.ipynb                         Metadata agent training
├── URL_Agent.ipynb                              URL agent training
│
├── dataset_handling.py                          Multi-source corpus loaders
├── feature_extraction.py                        URL + metadata feature engineering
├── metadata_agent.py                            Random Forest metadata wrapper
├── url_agent.py                                 Random Forest URL wrapper
│
├── augmentation_cell.py                         Colab cell — augment training set
│                                                with PhishNChips + cybersectony +
│                                                hand-templated emails
│
└── synthetic_legit_emails.csv                   150 hand-templated modern legits
```

The earlier `gradio_demo_cell.py`, `generate_qr.py`, `main.py` and `dist.zip`
have been moved out of this repo: the deployable system is now the
**PhishLens** Chrome extension + FastAPI backend (linked above).

---

## Quick reproduction

The simplest way to reproduce the project end-to-end is to open the three
Colab notebooks in this order on a free T4 GPU runtime:

1. **`DistilBERT_Phishing_Text_Agent.ipynb`**
   Loads the MeAJOR Corpus from Hugging Face, fine-tunes DistilBERT for 3
   epochs, evaluates on a stratified 20% held-out test set, and saves the
   model.
2. **`URL_Agent.ipynb`** and **`Metadata_Agent.ipynb`**
   Build the multi-source EML corpus (Nazario + phishing\_pot + Enron Ham
   + Cisco Umbrella top-1m for legit URLs), train Random Forests, and
   report per-class metrics.

To reproduce the **out-of-distribution augmentation** described in §4.5.1
of the project report:

3. Insert the cell from `augmentation_cell.py` into the DistilBERT
   notebook immediately before the train/test split. It pulls
   `AreLit/PhishNChips`, `cybersectony/PhishingEmailDetectionv2.0`, and
   the local `synthetic_legit_emails.csv`, deduplicates, and replaces the
   training DataFrame. Re-run from that cell onwards.

To run the **deployable system** (Chrome extension + local backend +
real-time Gmail integration):

4. Head over to the **[PhishLens](https://github.com/Sonje03/PhishLens)**
   repository and follow its quickstart (Docker or manual install).

---

## Datasets

| Source                                            | Role                          | Count  |
|---------------------------------------------------|-------------------------------|-------:|
| zefang-liu/phishing-email-dataset (MeAJOR Corpus) | Baseline text agent training  | 18,650 |
| rf-peixoto/phishing\_pot                          | Real-world phishing samples   | varies |
| Nazario phishing corpus (filtered to ≥ 2022)      | Modern phishing baseline      | varies |
| SetFit/enron\_spam (Enron Ham)                    | Legitimate baseline           | varies |
| AreLit/PhishNChips                                | Modern workplace legits       | 1,333  |
| cybersectony/PhishingEmailDetectionv2.0           | Augmentation legits           | 11,322 |
| `synthetic_legit_emails.csv` (this repo)          | NG-domain hand-templated      | 150    |
| Cisco Umbrella top-1m                             | URL agent legit baseline      | 10,000 |

After deduplication, the augmented text corpus contains **29,555 emails**
(17,447 legitimate / 12,108 phishing, ratio 1.44).

---

## Trained model

The DistilBERT checkpoint (`model.safetensors`, ~268 MB) is **not stored in
this repo** — GitHub's per-file limit is 100 MB. It is hosted on Hugging
Face Hub: **`<TODO-handle>/phishlens-distilbert`** *(will be updated when
published)*.

The URL and metadata Random Forest `.pkl` artefacts are smaller and shipped
alongside the PhishLens backend image.

---

## Limitations

The system has four known limitations:

1. **English-only** — tokenizer and training corpora are English; no
   support for French / Hausa / Yoruba phishing at present.
2. **PDF attachments are ignored** — only `text/plain` and `text/html`
   parts of an `.eml` file are parsed. Emails whose substantive content
   lives in an attached PDF (lab reports, invoices, contracts) are
   effectively classified on an empty body. The PhishLens runtime
   mitigates this with a trusted-domain allowlist.
3. **Niche marketing / recruitment false positives** — promotional
   messages and specialised recruitment emails outside the augmentation
   distribution are still flagged.
4. **Static evaluation** — measured on held-out and qualitative test
   sets, not yet on a live email stream.

---

## License

This work is the property of Nile University of Nigeria.
The library has the right to make copies for educational purposes only.

---

## Citation

If you reference this work, please cite it as a B.Sc. Final Year Project
of the Department of Cybersecurity, Faculty of Computing, Nile University
of Nigeria, 2025/2026 academic session. Full author attribution is
available on request.
