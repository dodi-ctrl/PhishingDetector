# =====================================================
# GRADIO INFERENCE UI WITH LIME EXPLANATIONS
# Add this cell at the END of the DistilBERT notebook,
# AFTER the model has been trained and saved.
#
# Provides a public shareable URL valid for 72h via gradio.live.
# Perfect for the live defense demo on May 19.
# =====================================================

!pip install -q gradio lime

import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')   # avoid showing plots in notebook
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer
from email import message_from_string
from email.policy import default as email_default_policy

# ---------- Predict helpers ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

CLASS_NAMES = ['Safe', 'Phishing']

def predict_proba_for_lime(texts):
    """Returns numpy array of shape (n, 2) with [p_safe, p_phishing] per text."""
    all_probs = []
    BATCH = 16
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)

def parse_eml_bytes(raw_bytes):
    """Parse an .eml file into (subject, body) for the model."""
    try:
        msg = message_from_string(raw_bytes.decode('utf-8', errors='ignore'), policy=email_default_policy)
    except Exception:
        msg = message_from_string(raw_bytes.decode('latin-1', errors='ignore'), policy=email_default_policy)
    subject = str(msg.get('Subject', '') or '')
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == 'text/plain':
                try:
                    body = part.get_content()
                except Exception:
                    body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                break
        if not body:
            for part in msg.walk():
                if part.get_content_type() == 'text/html':
                    try:
                        body = part.get_content()
                    except Exception:
                        body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    break
    else:
        try:
            body = msg.get_content()
        except Exception:
            payload = msg.get_payload(decode=True)
            body = payload.decode('utf-8', errors='ignore') if payload else str(msg.get_payload())
    return subject, body

# ---------- Core inference ----------
explainer = LimeTextExplainer(class_names=CLASS_NAMES)

def classify_and_explain(text, num_features=10, num_samples=500):
    if not text or not text.strip():
        return ("⚠️ Please paste an email or upload an .eml file.", None, "No input.")

    # Predict
    probs = predict_proba_for_lime([text])[0]
    p_safe, p_phish = float(probs[0]), float(probs[1])
    pred_label = CLASS_NAMES[int(np.argmax(probs))]

    # LIME explanation
    exp = explainer.explain_instance(
        text,
        predict_proba_for_lime,
        num_features=num_features,
        num_samples=num_samples,
        labels=[1],   # explain w.r.t. Phishing class
    )

    # Build matplotlib figure (better looking than fig=exp.as_pyplot_figure() default)
    feats = exp.as_list(label=1)  # list of (token, weight) sorted by absolute weight desc
    tokens = [f for f, _ in feats]
    weights = [w for _, w in feats]
    colors = ['#d9534f' if w > 0 else '#5cb85c' for w in weights]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y_pos = np.arange(len(tokens))[::-1]   # top = strongest
    ax.barh(y_pos, weights, color=colors, edgecolor='black', linewidth=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tokens)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Contribution toward "Phishing" (red) vs "Safe" (green)')
    ax.set_title(f'LIME explanation — predicted: {pred_label} ({max(p_safe, p_phish)*100:.2f}%)')
    plt.tight_layout()

    # Verdict text
    if pred_label == 'Phishing':
        verdict = (
            f"🚨  **PHISHING**  —  confidence  **{p_phish*100:.2f}%**\n\n"
            f"P(Safe) = {p_safe*100:.2f}%   |   P(Phishing) = {p_phish*100:.2f}%\n\n"
            f"Top phishing-leaning tokens (red bars), top safe-leaning tokens (green bars):"
        )
    else:
        verdict = (
            f"✅  **SAFE**  —  confidence  **{p_safe*100:.2f}%**\n\n"
            f"P(Safe) = {p_safe*100:.2f}%   |   P(Phishing) = {p_phish*100:.2f}%\n\n"
            f"The model considers this email legitimate. LIME shows the most influential tokens below."
        )

    # Build readable token detail string
    detail = "\n".join([
        f"  {token:25s}  →  {weight:+.4f}  ({'phishing' if weight > 0 else 'safe'})"
        for token, weight in feats
    ])

    return verdict, fig, detail

def classify_eml(file_obj, num_features, num_samples):
    """Wrapper for .eml upload."""
    if file_obj is None:
        return ("⚠️ No file provided.", None, "")
    with open(file_obj.name, 'rb') as f:
        raw = f.read()
    subject, body = parse_eml_bytes(raw)
    full_text = f"Subject: {subject}\n\n{body}".strip()
    verdict, fig, detail = classify_and_explain(full_text, int(num_features), int(num_samples))
    return f"**Subject parsed:** {subject}\n\n{verdict}", fig, detail


def classify_batch(files, num_features, num_samples, progress=gr.Progress()):
    """Classify a batch of .eml files and return a results table + downloadable CSV."""
    if not files:
        return None, "⚠️ No files provided. Drop one or more .eml files in the box above.", None

    rows = []
    n = len(files)
    progress(0, desc=f"Starting batch of {n} email(s)...")

    for i, file_obj in enumerate(files):
        fname = Path(file_obj.name).name
        progress((i + 0.5) / n, desc=f"[{i+1}/{n}] {fname[:60]}")

        try:
            with open(file_obj.name, 'rb') as f:
                raw = f.read()
            subject, body = parse_eml_bytes(raw)
            full_text = f"Subject: {subject}\n\n{body}".strip()

            if not full_text or len(full_text) < 5:
                rows.append({
                    "File": fname,
                    "Subject": subject[:80] if subject else "(empty)",
                    "Verdict": "ERROR",
                    "Confidence": "—",
                    "P(Safe)": "—",
                    "P(Phishing)": "—",
                    "Top phishing tokens": "(empty body)",
                    "Top safe tokens": "—",
                })
                continue

            # Classify
            probs = predict_proba_for_lime([full_text])[0]
            p_safe, p_phish = float(probs[0]), float(probs[1])
            pred_label = CLASS_NAMES[int(np.argmax(probs))]
            confidence = max(p_safe, p_phish)

            # LIME (use lower sample count by default — speed matters here)
            exp = explainer.explain_instance(
                full_text,
                predict_proba_for_lime,
                num_features=int(num_features),
                num_samples=int(num_samples),
                labels=[1],
            )
            feats = exp.as_list(label=1)
            phishing_tokens = [t for t, w in feats if w > 0][:3]
            safe_tokens = [t for t, w in feats if w < 0][:3]

            rows.append({
                "File": fname,
                "Subject": subject[:80] + ("..." if len(subject) > 80 else ""),
                "Verdict": pred_label,
                "Confidence": f"{confidence*100:.2f}%",
                "P(Safe)": f"{p_safe*100:.2f}%",
                "P(Phishing)": f"{p_phish*100:.2f}%",
                "Top phishing tokens": ", ".join(phishing_tokens) or "—",
                "Top safe tokens": ", ".join(safe_tokens) or "—",
            })
        except Exception as e:
            rows.append({
                "File": fname,
                "Subject": f"(parse error: {type(e).__name__})",
                "Verdict": "ERROR",
                "Confidence": "—",
                "P(Safe)": "—",
                "P(Phishing)": "—",
                "Top phishing tokens": str(e)[:60],
                "Top safe tokens": "—",
            })

    progress(1.0, desc="Done.")
    df = pd.DataFrame(rows)

    # Summary block
    n_total    = len(rows)
    n_phishing = sum(1 for r in rows if r["Verdict"] == "PHISHING")
    n_safe     = sum(1 for r in rows if r["Verdict"] == "SAFE")
    n_error    = sum(1 for r in rows if r["Verdict"] == "ERROR")

    summary = (
        f"### 📊 Batch results — {n_total} email{'s' if n_total != 1 else ''} processed\n\n"
        f"- 🚨 **PHISHING**: {n_phishing}  ({n_phishing/n_total*100:.0f}%)\n"
        f"- ✅ **SAFE**:      {n_safe}  ({n_safe/n_total*100:.0f}%)\n"
        f"- ❌ **ERROR**:     {n_error}  ({n_error/n_total*100:.0f}%)\n"
    )

    # Save CSV for download
    csv_path = "/tmp/batch_results.csv"
    df.to_csv(csv_path, index=False)

    return df, summary, csv_path

# ---------- Gradio UI ----------
EXAMPLES = [
    ["""Subject: Your account has been suspended

Dear Customer,

We have detected suspicious activity on your account. Your account has been temporarily suspended for your protection.

Please verify your identity immediately by clicking the link below:
http://secure-bank-verify.com/login

If you do not verify within 24 hours, your account will be permanently closed.

Thank you,
Security Team"""],
    ["""Subject: Notification of Second Semester Course Registration

Dear Student,

This is to notify you that the Second Semester Course Registration for the 2025/2026 academic session is now open.

Kindly log into the student portal to complete your registration before the deadline.

Best regards,
Office of Academic Affairs
Nile University of Nigeria"""],
    ["""Subject: Project meeting Friday 3pm

Hi team,

Quick reminder we have the project sync Friday at 3pm in the small conference room.
Please come prepared with status updates on your work streams.

Thanks,
Sarah"""],
]

with gr.Blocks(title="Smart Phishing Detector — DistilBERT + LIME") as demo:
    gr.Markdown(
        """
        # 🛡️ Smart Phishing Detector
        ### CYB 499 Final Year Project — Nile University of Nigeria

        Paste any email text *or* upload a `.eml` file. The DistilBERT model classifies it as **Safe** or **Phishing**,
        and LIME explains *which words* drove the decision.
        """
    )

    with gr.Tabs():
        with gr.TabItem("📝 Paste email text"):
            with gr.Row():
                with gr.Column(scale=1):
                    text_in = gr.Textbox(
                        label="Email content",
                        placeholder="Paste the full email here (Subject + body)...",
                        lines=14,
                    )
                    nf = gr.Slider(5, 20, value=10, step=1, label="Number of LIME features")
                    ns = gr.Slider(200, 1500, value=500, step=100, label="LIME samples (quality vs speed)")
                    btn = gr.Button("🔍 Analyze", variant="primary")
                    gr.Examples(examples=EXAMPLES, inputs=[text_in])
                with gr.Column(scale=1):
                    verdict_out = gr.Markdown(label="Verdict")
                    fig_out = gr.Plot(label="LIME explanation")
                    detail_out = gr.Code(label="Token contributions", language="markdown")
            btn.click(fn=classify_and_explain, inputs=[text_in, nf, ns], outputs=[verdict_out, fig_out, detail_out])

        with gr.TabItem("📎 Upload .eml file"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_in = gr.File(label=".eml file", file_types=['.eml'])
                    nf2 = gr.Slider(5, 20, value=10, step=1, label="Number of LIME features")
                    ns2 = gr.Slider(200, 1500, value=500, step=100, label="LIME samples")
                    btn2 = gr.Button("🔍 Analyze .eml", variant="primary")
                with gr.Column(scale=1):
                    verdict_out2 = gr.Markdown(label="Verdict")
                    fig_out2 = gr.Plot(label="LIME explanation")
                    detail_out2 = gr.Code(label="Token contributions", language="markdown")
            btn2.click(fn=classify_eml, inputs=[file_in, nf2, ns2], outputs=[verdict_out2, fig_out2, detail_out2])

        with gr.TabItem("📚 Batch .eml upload"):
            gr.Markdown(
                "Drop **multiple `.eml` files** at once. Each will be classified "
                "and explained; results appear as a sortable table and as a "
                "downloadable CSV."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    files_in = gr.File(
                        label="Drop multiple .eml files",
                        file_count="multiple",
                        file_types=['.eml'],
                    )
                    nf3 = gr.Slider(5, 15, value=8, step=1,
                                    label="LIME features per email")
                    ns3 = gr.Slider(100, 800, value=250, step=50,
                                    label="LIME samples (lower = faster batches)")
                    btn3 = gr.Button("🚀 Analyze all", variant="primary")
                    summary_out = gr.Markdown()
                    csv_out = gr.File(label="📥 Download results CSV",
                                       interactive=False)
                with gr.Column(scale=2):
                    df_out = gr.Dataframe(
                        label="Per-email results",
                        wrap=True,
                        interactive=False,
                        headers=[
                            "File", "Subject", "Verdict", "Confidence",
                            "P(Safe)", "P(Phishing)",
                            "Top phishing tokens", "Top safe tokens",
                        ],
                    )
            btn3.click(
                fn=classify_batch,
                inputs=[files_in, nf3, ns3],
                outputs=[df_out, summary_out, csv_out],
            )

    gr.Markdown(
        """
        ---
        **About this demo.** The model is a fine-tuned DistilBERT trained on a curated multi-source corpus
        (Nazario, MeAJOR, phishing_pot, Enron Ham, PhishNChips, cybersectony PhishingEmailDetectionv2.0)
        plus 150 hand-templated modern legitimate emails. LIME (Ribeiro et al., 2016) is used to expose
        which tokens drove each prediction.
        """
    )

# Launch with public URL (valid 72h, perfect for the defense)
demo.launch(share=True, debug=False)
