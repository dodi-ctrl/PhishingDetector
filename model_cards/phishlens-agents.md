---
license: mit
library_name: sklearn
tags:
  - phishing-detection
  - email-security
  - random-forest
  - tabular-classification
  - skops
pipeline_tag: tabular-classification
---

# PhishLens agents: URL and metadata Random Forests

Two scikit-learn Random Forest classifiers used by
[PhishLens](https://github.com/AnzouK/PhishLens), a Chrome extension and
FastAPI backend that detects phishing emails in Gmail and explains its
verdicts. They sit next to a fine-tuned DistilBERT text classifier
([AnzouKiona/phishlens-distilbert](https://huggingface.co/AnzouKiona/phishlens-distilbert));
the three scores are fused and then adjusted with SPF / DKIM / DMARC
results and URL threat intelligence.

| File | Agent | Input | Held-out F1 | ROC-AUC |
| --- | --- | --- | --- | --- |
| `url_agent.skops` | URL agent | 23 features of the links in the email | 0.9911 | 0.9969 |
| `metadata_agent.skops` | Metadata agent | 37 header features (SPF, DKIM, DMARC, Reply-To, routing, subject) | 0.9994 | 1.0000 |

The `.joblib` files are the original exports and are kept for
reproducibility only. Use the `.skops` files: they load without pickle.

## Use

Each file holds a dict with `model` (RandomForestClassifier), `scaler`
(StandardScaler), `feature_names` and training metadata. Features must
be produced by `FeatureExtractor` from
[feature_extraction.py](https://github.com/dodi-ctrl/PhishingDetector/blob/main/feature_extraction.py),
in the order given by `feature_names`.

```python
from huggingface_hub import hf_hub_download
import skops.io as sio

path = hf_hub_download("AnzouKiona/phishlens-agents", "url_agent.skops")
untrusted = sio.get_untrusted_types(file=path)
# Only accept scikit-learn / NumPy / SciPy / builtins types.
assert all(t.startswith(("sklearn.", "numpy.", "scipy.", "builtins.")) for t in untrusted)
state = sio.load(path, trusted=untrusted)

X = state["scaler"].transform(features_df[state["feature_names"]].values)
p_phishing = state["model"].predict_proba(X)[:, 1]
```

The PhishLens backend does exactly this at startup
([agent_io.py](https://github.com/AnzouK/PhishLens/blob/main/backend/agent_io.py)).

## Training data

Trained in the [PhishingDetector](https://github.com/dodi-ctrl/PhishingDetector)
notebooks on a multi-source `.eml` corpus: rf-peixoto/phishing_pot and
the Nazario corpus (phishing), Enron ham (legitimate), and the Cisco
Umbrella top-1m list for legitimate URLs. Trained with scikit-learn
1.6.1; the skops files were produced from those models under
scikit-learn 1.9.1 and reproduce their predictions exactly.

## Limitations

- **Upper-bound scores.** Phishing and legitimate samples come from
  different corpora and years, so header artefacts make the classes
  easier to separate than in a real inbox, especially for the metadata
  agent.
- **English-centric data** and offline evaluation only; no live-inbox
  measurement yet.
- The agents are one signal among several in PhishLens and are not
  meant to be used alone to block mail.

## Citation

B.Sc. final-year project, Department of Cybersecurity, Nile University
of Nigeria, 2025/2026. Released under the MIT License.
