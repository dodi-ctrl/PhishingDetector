"""
Simplified Feature Extraction for Phishing Detection
Extracts only the most impactful features in a clean, elegant format.

Author: Ifanyi Uche Henry
Project: Social Engineering Attacks (Phishing) Detection using NLP
"""

import re
import pandas as pd
import numpy as np
from email import policy
from email.parser import BytesParser
from urllib.parse import urlparse


class FeatureExtractor:
    """
    Streamlined feature extraction focusing on high-impact signals.
    Returns pandas DataFrame for direct ML pipeline integration.
    """
    
    def __init__(self):
        self.urgent_words = ['urgent', 'immediate', 'act now', 'verify', 'suspend', 
                            'confirm', 'security alert', 'limited time']
        self.sensitive_requests = ['password', 'social security', 'ssn', 
                                  'credit card', 'bank account']
        self.suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz']
        
    def extract_text_features(self, text):
        """Extract key text-based phishing indicators."""
        text_lower = text.lower() if text else ""
        
        return {
            'text_length': len(text),
            'uppercase_ratio': sum(c.isupper() for c in text) / max(len(text), 1),
            'exclamation_count': text.count('!'),
            'urgent_word_count': sum(word in text_lower for word in self.urgent_words),
            'requests_sensitive_info': int(any(req in text_lower for req in self.sensitive_requests)),
            'has_click_here': int('click here' in text_lower),
            'generic_greeting': int(any(g in text_lower for g in ['dear customer', 'dear user', 'valued customer'])),
            'contains_currency': int(bool(re.search(r'\$\d+', text))),
            'url_count': len(re.findall(r'https?://', text))
        }
    
    def extract_url_features(self, text):
        """Extract URL-based phishing indicators."""
        urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', text)
        
        if not urls:
            return {
                'has_url': 0,
                'domain_length': 0,
                'has_ip_address': 0,
                'suspicious_tld': 0,
                'uses_https': 0,
                'excessive_subdomains': 0,
                'url_has_at_symbol': 0,
                'shortened_url': 0
            }
        
        url = urls[0]  # Analyze primary URL
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        return {
            'has_url': 1,
            'domain_length': len(domain),
            'has_ip_address': int(bool(re.match(r'\d+\.\d+\.\d+\.\d+', domain))),
            'suspicious_tld': int(any(domain.endswith(tld) for tld in self.suspicious_tlds)),
            'uses_https': int(parsed.scheme == 'https'),
            'excessive_subdomains': int(domain.count('.') > 3),
            'url_has_at_symbol': int('@' in url),
            'shortened_url': int(any(s in domain for s in ['bit.ly', 'tinyurl', 't.co']))
        }
    
    def extract_metadata_features(self, email_content):
        """Extract email header and authentication features."""
        try:
            if isinstance(email_content, str):
                email_content = email_content.encode()
            msg = BytesParser(policy=policy.default).parsebytes(email_content)
        except:
            return self._default_metadata_features()
        
        # Extract domains
        from_domain = self._get_domain(msg.get('From', ''))
        reply_domain = self._get_domain(msg.get('Reply-To', ''))
        
        # Authentication results
        spf = msg.get('Received-SPF', '').lower()
        auth = msg.get('Authentication-Results', '').lower()
        subject = msg.get('Subject', '')
        
        return {
            'sender_domain_length': len(from_domain),
            'reply_to_mismatch': int(reply_domain != '' and reply_domain != from_domain),
            'spf_fail': int('fail' in spf),
            'dkim_fail': int('dkim=fail' in auth),
            'dmarc_fail': int('dmarc=fail' in auth),
            'auth_none': int(not spf and 'dkim' not in auth),
            'subject_all_caps': int(subject.isupper() and len(subject) > 5),
            'subject_urgent_keywords': sum(word in subject.lower() for word in self.urgent_words),
            'high_priority': int('1' in msg.get('X-Priority', '')),
            'num_received_headers': len(msg.get_all('Received', []))
        }
    
    def extract_all_features(self, email_content, text_content=None):
        """
        Extract all features and return as single DataFrame row.
        
        Parameters:
        -----------
        email_content : str or bytes
            Full email with headers
        text_content : str, optional
            Email body text (extracted if not provided)
            
        Returns:
        --------
        pd.DataFrame : Single row with all features
        """
        # Extract text if not provided
        if text_content is None:
            try:
                if isinstance(email_content, str):
                    msg = BytesParser(policy=policy.default).parsebytes(email_content.encode())
                else:
                    msg = BytesParser(policy=policy.default).parsebytes(email_content)
                text_content = self._get_email_body(msg)
            except:
                text_content = str(email_content)
        
        # Combine all features
        features = {}
        features.update(self.extract_text_features(text_content))
        features.update(self.extract_url_features(text_content))
        features.update(self.extract_metadata_features(email_content))
        
        return pd.DataFrame([features])
    
    def extract_batch(self, email_list, text_list=None):
        """
        Extract features from multiple emails efficiently.
        
        Parameters:
        -----------
        email_list : list
            List of email contents
        text_list : list, optional
            List of email body texts
            
        Returns:
        --------
        pd.DataFrame : Feature matrix for all emails
        """
        if text_list is None:
            text_list = [None] * len(email_list)
        
        features_list = []
        for email, text in zip(email_list, text_list):
            features = self.extract_all_features(email, text)
            features_list.append(features)
        
        return pd.concat(features_list, ignore_index=True)
    
    # Helper methods
    def _get_domain(self, email_string):
        """Extract domain from email header."""
        match = re.search(r'[\w\.-]+@([\w\.-]+)', email_string)
        return match.group(1) if match else ''
    
    def _get_email_body(self, msg):
        """Extract text body from email."""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == 'text/plain':
                    return part.get_content()
        return msg.get_content() if hasattr(msg, 'get_content') else ''
    
    def _default_metadata_features(self):
        """Return default metadata when parsing fails."""
        return {
            'sender_domain_length': 0,
            'reply_to_mismatch': 0,
            'spf_fail': 0,
            'dkim_fail': 0,
            'dmarc_fail': 0,
            'auth_none': 1,
            'subject_all_caps': 0,
            'subject_urgent_keywords': 0,
            'high_priority': 0,
            'num_received_headers': 0
        }


# Quick test
if __name__ == "__main__":
    extractor = FeatureExtractor()
    
    # Sample phishing email
    sample = """From: admin@paypa1-secure.tk
To: victim@example.com
Subject: URGENT ACTION REQUIRED!!!
Received-SPF: fail
X-Priority: 1

Dear Customer,

Your account will be SUSPENDED! Click here immediately:
http://192.168.1.1/verify?token=12345

Enter your password and credit card to confirm.
"""
    
    # Extract features
    features = extractor.extract_all_features(sample)
    
    print("Features extracted:")
    print(features.T)
    print(f"\nTotal features: {len(features.columns)}")
