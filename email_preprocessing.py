import os
import mailbox
import re
from email import policy
from email.parser import BytesParser
from email.utils import parsedate_to_datetime


def _extract_body(msg):
    """Return plain text from a parsed email, falling back to stripped HTML."""
    plain_text = None
    html_text = None

    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == 'text/plain' and plain_text is None:
                plain_text = part.get_content()
            elif ct == 'text/html' and html_text is None:
                html_text = part.get_content()
    else:
        ct = msg.get_content_type()
        if ct == 'text/plain':
            plain_text = msg.get_content()
        elif ct == 'text/html':
            html_text = msg.get_content()

    if plain_text:
        return plain_text
    if html_text:
        return re.sub(r'<[^>]+>', ' ', html_text)
    return ''


def preprocess_email(email):
    """
    Extract features from an email.

    Accepts either:
    - A dict with 'body', 'from', and optional 'label' keys
    - A raw RFC 2822 email as a str or bytes (multipart MIME supported)
    """
    urgency_words = [
        'urgent', 'immediately', 'verify', 'suspended', 'confirm',
        'action required', 'act now', 'limited time', 'security alert',
        'unusual activity', 'locked', 'expired', 'validate', 'authenticate',
        'click here', 'update your', 'verify your account',
    ]

    if isinstance(email, (str, bytes)):
        # Raw RFC 2822 input — parse headers and extract body via MIME walker
        raw_bytes = email.encode() if isinstance(email, str) else email
        msg = BytesParser(policy=policy.default).parsebytes(raw_bytes)
        raw_text = _extract_body(msg)
        sender = msg.get('From', '')
        label = None
    else:
        # Dict input (plain body string already provided)
        raw_text = email.get('body', '')
        sender = email.get('from', '')
        label = email.get('label', None)

    # Strip HTML tags from whatever body we ended up with
    text = re.sub(r'<[^>]+>', ' ', raw_text)
    text = text.lower().strip()

    # Extract URLs — avoid capturing trailing punctuation with [^\s<>"']+
    urls = re.findall(r'https?://[^\s<>"\']+', raw_text)

    urgency_count = sum(text.count(word) for word in urgency_words)

    features = {
        'text': text,
        'text_length': len(text),
        'url_count': len(urls),
        'urgency_score': urgency_count,
        'has_ip_url': any(re.search(r'\d+\.\d+\.\d+\.\d+', url) for url in urls),
        'sender': sender,
        'label': label,
    }

    return features


def preprocess_eml_file(path, label=None):
    """
    Read a single .eml file from disk and return preprocess_email() output.
    Adds 'source_path' and (optional) 'label' fields.
    """
    with open(path, 'rb') as f:
        raw = f.read()
    result = preprocess_email(raw)
    result['source_path'] = path
    if label is not None:
        result['label'] = label
    return result


def preprocess_eml_directory(directory, label=None, recursive=True):
    """
    Walk a directory, preprocess every .eml file, and return a list of
    feature dicts. Useful for bulk-processing phishing_pot exports.
    """
    results = []
    if not os.path.isdir(directory):
        print(f"Warning: directory not found: {directory}")
        return results

    walker = os.walk(directory) if recursive else [(directory, [], os.listdir(directory))]
    for root, _, files in walker:
        for fname in files:
            if fname.lower().endswith('.eml'):
                fpath = os.path.join(root, fname)
                try:
                    results.append(preprocess_eml_file(fpath, label=label))
                except Exception as e:
                    print(f"  Skipping {fpath}: {e}")
    return results


def preprocess_mbox(mbox_path, label=None, after_year=None):
    """
    Iterate an mbox file (e.g. Nazario phishing3.mbox) and return a list of
    preprocess_email() outputs. Optionally filter to messages from
    after_year onwards (using the Date header).
    """
    results = []
    if not os.path.isfile(mbox_path):
        print(f"Warning: mbox not found: {mbox_path}")
        return results

    for msg in mailbox.mbox(mbox_path):
        try:
            if after_year is not None:
                try:
                    if parsedate_to_datetime(msg.get('Date', '')).year < after_year:
                        continue
                except Exception:
                    pass  # include if Date is unparseable
            features = preprocess_email(msg.as_bytes())
            if label is not None:
                features['label'] = label
            results.append(features)
        except Exception:
            continue
    return results


if __name__ == "__main__":
    # Test 1: Phishing email (dict)
    print("TEST 1: Phishing Email")
    print("-" * 50)
    phishing_email = {
        'body': '<p>URGENT! Your account is suspended. Verify at http://fake-bank.com</p>',
        'from': 'scammer@evil.com'
    }
    result1 = preprocess_email(phishing_email)
    print(f"Original: {phishing_email['body']}")
    print(f"Cleaned text: {result1['text']}")
    print(f"Text length: {result1['text_length']}")
    print(f"URLs found: {result1['url_count']}")
    print(f"Urgency score: {result1['urgency_score']}")
    print(f"Has IP in URL: {result1['has_ip_url']}")
    print()

    # Test 2: Legitimate email (dict)
    print("TEST 2: Legitimate Email")
    print("-" * 50)
    legit_email = {
        'body': 'Hi, the meeting is at 3pm. Here is the link: https://zoom.us/meeting',
        'from': 'colleague@company.com'
    }
    result2 = preprocess_email(legit_email)
    print(f"Original: {legit_email['body']}")
    print(f"Cleaned text: {result2['text']}")
    print(f"Text length: {result2['text_length']}")
    print(f"URLs found: {result2['url_count']}")
    print(f"Urgency score: {result2['urgency_score']}")
    print(f"Has IP in URL: {result2['has_ip_url']}")
    print()

    # Test 3: Phishing with IP address in URL (dict)
    print("TEST 3: Phishing with IP Address")
    print("-" * 50)
    ip_phishing = {
        'body': 'Click here immediately: http://192.168.1.1/verify',
        'from': 'fake@phish.com'
    }
    result3 = preprocess_email(ip_phishing)
    print(f"Original: {ip_phishing['body']}")
    print(f"Cleaned text: {result3['text']}")
    print(f"URLs found: {result3['url_count']}")
    print(f"Urgency score: {result3['urgency_score']}")
    print(f"Has IP in URL: {result3['has_ip_url']}")
    print()

    # Test 4: Empty email (dict)
    print("TEST 4: Empty Email")
    print("-" * 50)
    empty_email = {
        'body': '',
        'from': 'test@test.com'
    }
    result4 = preprocess_email(empty_email)
    print(f"Text length: {result4['text_length']}")
    print(f"URLs found: {result4['url_count']}")
    print(f"Urgency score: {result4['urgency_score']}")
    print()

    # Test 5: Raw RFC 2822 multipart MIME email
    print("TEST 5: Raw Multipart MIME Email")
    print("-" * 50)
    raw_email = b"""\
From: attacker@evil.com
To: victim@example.com
Subject: Urgent: Verify your account
MIME-Version: 1.0
Content-Type: multipart/alternative; boundary="boundary123"

--boundary123
Content-Type: text/plain; charset=utf-8

Verify your account immediately: http://evil.com/verify
--boundary123
Content-Type: text/html; charset=utf-8

<html><body><p>Verify your <b>account</b> immediately:
<a href="http://evil.com/verify">click here</a></p></body></html>
--boundary123--
"""
    result5 = preprocess_email(raw_email)
    print(f"Sender: {result5['sender']}")
    print(f"Cleaned text: {result5['text']!r}")
    print(f"URLs found: {result5['url_count']}")
    print(f"Urgency score: {result5['urgency_score']}")
    print()

    print("=" * 50)
    print("ALL TESTS COMPLETED!")
    print("=" * 50)
