"""
FEATURE EXTRACTION FOR PHISHING DETECTION

"""

import re
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from collections import Counter
"""For accurate TLD extraction, optional but recommended"""
try:
    import tldextract 
    HAS_TLDEXTRACT = True
except ImportError:
    HAS_TLDEXTRACT = False
import math



# TEXT FEATURES


def extract_text_features(text):
    """Extract text-based features"""
    if not text:
        return {
            'text_length': 0, 'word_count': 0, 'avg_word_len': 0,
            'uppercase_ratio': 0, 'digit_ratio': 0,
            'urgency_score': 0, 'threat_score': 0,
            'sentiment_score': 0
        }
    
    words = text.split()
    
    # Keyword matching
    urgency = ['urgent', 'immediately', 'asap', 'act now', 'expires', 'limited time']
    threat = ['suspended', 'terminated', 'locked', 'unauthorized', 'compromised']
    
    return {
        'text_length': len(text),
        'word_count': len(words),
        'avg_word_len': sum(len(w) for w in words) / len(words) if words else 0,
        'uppercase_ratio': sum(c.isupper() for c in text) / len(text) if text else 0,
        'digit_ratio': sum(c.isdigit() for c in text) / len(text) if text else 0,
        'urgency_score': sum(text.lower().count(w) for w in urgency),
        'threat_score': sum(text.lower().count(w) for w in threat),
        'sentiment_score': text.count('!') - text.count('?'),  # Simple sentiment proxy
    }



# URL FEATURES


def extract_url_features(urls):
    """Extract URL-based features"""
    if not urls:
        return {
            'url_count': 0, 'avg_url_len': 0, 'has_ip': 0,
            'has_shortener': 0, 'suspicious_tld': 0,
            'https_ratio': 0, 'avg_entropy': 0
        }
    
    shorteners = ['bit.ly', 'tinyurl', 'goo.gl', 't.co', 'ow.ly']
    bad_tlds = ['tk', 'ml', 'ga', 'cf', 'gq', 'xyz']
    
    # Calculate entropy
    def entropy(s):
        p = Counter(s)
        l = len(s)
        return -sum((c/l) * math.log2(c/l) for c in p.values())
    
    # Check suspicious TLD
    if HAS_TLDEXTRACT:
        suspicious_tld = int(any(tldextract.extract(u).suffix in bad_tlds for u in urls))
    else:
        # Fallback: simple string check
        suspicious_tld = int(any(any(u.endswith(f'.{tld}') for tld in bad_tlds) for u in urls))
    
    features = {
        'url_count': len(urls),
        'avg_url_len': sum(len(u) for u in urls) / len(urls),
        'has_ip': int(any(re.search(r'\d+\.\d+\.\d+\.\d+', u) for u in urls)),
        'has_shortener': int(any(s in u for u in urls for s in shorteners)),
        'suspicious_tld': suspicious_tld,
        'https_ratio': sum(u.startswith('https') for u in urls) / len(urls),
        'avg_entropy': sum(entropy(u) for u in urls) / len(urls)
    }
    
    return features


# METADATA FEATURES

def extract_metadata_features(email):
    """Extract metadata features"""
    sender = email.get('from', '').lower()
    reply_to = email.get('reply-to', '').lower()
    subject = email.get('subject', '')
    
    # Extract domains
    sender_domain = re.search(r'@([a-z0-9.-]+)', sender)
    reply_domain = re.search(r'@([a-z0-9.-]+)', reply_to) if reply_to else None
    
    # Check domain mismatch only if both exist
    domain_mismatch = 0
    if sender_domain and reply_domain:
        domain_mismatch = int(sender_domain.group(1) != reply_domain.group(1))
    
    return {
        'domain_mismatch': domain_mismatch,
        'subject_len': len(subject),
        'subject_caps_ratio': sum(c.isupper() for c in subject) / len(subject) if subject else 0,
        'urgent_subject': int(any(w in subject.lower() for w in ['urgent', 'verify', 'suspended'])),
        'excessive_punct': int(subject.count('!') > 1 or subject.count('?') > 1)
    }



# MAIN PIPELINE


def extract_all_features(email):
    """Complete feature extraction pipeline"""
    # Clean HTML
    raw_body = email.get('body', '')
    clean_text = BeautifulSoup(raw_body, 'html.parser').get_text()
    clean_text = clean_text.lower().strip()
    
    # Extract URLs
    urls = re.findall(r'http[s]?://[^\s<>"{}|\\^`\[\]]+', raw_body)
    
    # Get all features
    features = {
        **extract_text_features(clean_text),
        **extract_url_features(urls),
        **extract_metadata_features(email)
    }
    
    return features


def process_dataset(emails_list):
    """Process multiple emails into DataFrame"""
    features_list = []
    
    for email in emails_list:
        try:
            features = extract_all_features(email)
            features['label'] = email.get('label', None)
            features_list.append(features)
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    return pd.DataFrame(features_list)



# TEST


if __name__ == "__main__":
    # Test emails
    phishing = {
        'body': '<p>URGENT! Account suspended. Verify at http://192.168.1.1/verify NOW!</p>',
        'from': 'alert@bank-security.tk',
        'reply-to': 'scam@evil.com',
        'subject': 'URGENT: Verify Account!',
        'label': 1
    }
    
    legitimate = {
        'body': 'Hi, meeting at 3pm. Zoom link: https://zoom.us/j/123',
        'from': 'john@company.com',
        'subject': 'Re: Meeting',
        'label': 0
    }
    
    # Extract features
    print("PHISHING EMAIL FEATURES:")
    p_features = extract_all_features(phishing)
    for k, v in p_features.items():
        print(f"  {k}: {v}")
    
    print("\nLEGITIMATE EMAIL FEATURES:")
    l_features = extract_all_features(legitimate)
    for k, v in l_features.items():
        print(f"  {k}: {v}")
    
    # Process as dataset
    print("\nDATASET PROCESSING:")
    df = process_dataset([phishing, legitimate])
    print(df)
    print("\n Done!")
