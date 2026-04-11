"""
Feature Extraction Module for Phishing Detection
Extracts text, URL, and metadata features from email content.

Author: Udodinachukwu Nwoke
Project: Phishing Detection using NLP
"""

import re
import string
from urllib.parse import urlparse
from email import policy
from email.parser import BytesParser
import numpy as np
from collections import Counter


class FeatureExtractor:
    """
    Comprehensive feature extraction for phishing detection.
    Extracts features from text content, URLs, and email metadata.
    """
    
    def __init__(self):
        # Phishing keywords and patterns
        self.urgent_keywords = [
            'urgent', 'immediate', 'action required', 'verify', 'suspend',
            'limited time', 'act now', 'confirm', 'click here', 'update',
            'security alert', 'unusual activity', 'locked', 'expired',
            'validate', 'authenticate', 'prize', 'winner', 'congratulations',
            'claim', 'refund', 'suspended', 'verify your account'
        ]
        
        self.financial_keywords = [
            'bank', 'credit card', 'account', 'payment', 'transaction',
            'paypal', 'western union', 'wire transfer', 'social security',
            'tax', 'irs', 'refund', 'invoice', 'billing'
        ]
        
        self.threat_keywords = [
            'legal action', 'lawsuit', 'suspended', 'terminated', 'blocked',
            'restricted', 'violation', 'unauthorized', 'fraud', 'illegal'
        ]
        
        # Common legitimate domains (for comparison)
        self.legitimate_domains = [
            'google.com', 'microsoft.com', 'apple.com', 'amazon.com',
            'facebook.com', 'linkedin.com', 'twitter.com', 'instagram.com'
        ]
    
    def extract_text_features(self, text_content):
        """
        Extract features from email text content.
        
        Parameters:
        -----------
        text_content : str
            The body text of the email
            
        Returns:
        --------
        dict : Dictionary containing text-based features
        """
        features = {}
        
        try:
            if not text_content or not isinstance(text_content, str):
                text_content = ""
            
            text_lower = text_content.lower()
            text_clean = text_content.strip()
            
            # 1. BASIC TEXT STATISTICS
            features['text_length'] = len(text_content)
            features['word_count'] = len(text_content.split())
            features['avg_word_length'] = (
                np.mean([len(word) for word in text_content.split()])
                if features['word_count'] > 0 else 0
            )
            features['sentence_count'] = len(re.findall(r'[.!?]+', text_content))
            
            # 2. CHARACTER ANALYSIS
            features['uppercase_ratio'] = (
                sum(1 for c in text_content if c.isupper()) / len(text_content)
                if len(text_content) > 0 else 0
            )
            features['digit_ratio'] = (
                sum(1 for c in text_content if c.isdigit()) / len(text_content)
                if len(text_content) > 0 else 0
            )
            features['special_char_ratio'] = (
                sum(1 for c in text_content if c in string.punctuation) / len(text_content)
                if len(text_content) > 0 else 0
            )
            
            # 3. PUNCTUATION ANALYSIS
            features['exclamation_count'] = text_content.count('!')
            features['question_count'] = text_content.count('?')
            features['excessive_punctuation'] = int(
                features['exclamation_count'] > 3 or 
                text_content.count('!!') > 0
            )
            
            # 4. KEYWORD ANALYSIS
            features['urgent_keyword_count'] = sum(
                1 for keyword in self.urgent_keywords 
                if keyword in text_lower
            )
            features['financial_keyword_count'] = sum(
                1 for keyword in self.financial_keywords 
                if keyword in text_lower
            )
            features['threat_keyword_count'] = sum(
                1 for keyword in self.threat_keywords 
                if keyword in text_lower
            )
            
            # 5. SUSPICIOUS PATTERNS
            features['has_click_here'] = int('click here' in text_lower)
            features['has_verify_account'] = int('verify' in text_lower and 'account' in text_lower)
            features['has_act_now'] = int('act now' in text_lower or 'act immediately' in text_lower)
            features['has_limited_time'] = int('limited time' in text_lower)
            features['has_congratulations'] = int('congratulations' in text_lower)
            
            # 6. PERSONAL INFORMATION REQUESTS
            features['requests_password'] = int('password' in text_lower)
            features['requests_ssn'] = int('social security' in text_lower or 'ssn' in text_lower)
            features['requests_credit_card'] = int('credit card' in text_lower)
            features['requests_bank_info'] = int(
                'bank account' in text_lower or 
                'routing number' in text_lower or
                'account number' in text_lower
            )
            
            # 7. GREETING ANALYSIS
            generic_greetings = ['dear customer', 'dear user', 'dear member', 
                               'dear account holder', 'valued customer']
            features['generic_greeting'] = int(
                any(greeting in text_lower for greeting in generic_greetings)
            )
            
            # 8. SPELLING AND GRAMMAR (simple heuristics)
            # Count common misspellings
            common_misspellings = ['recieve', 'occured', 'seperate', 'untill', 'payed']
            features['misspelling_count'] = sum(
                1 for word in common_misspellings if word in text_lower
            )
            
            # 9. CURRENCY AND MONEY MENTIONS
            features['contains_currency'] = int(bool(re.search(r'\$\d+|\d+\s*dollars|\d+\s*USD', text_content)))
            features['large_sum_mentioned'] = int(bool(re.search(r'\$\d{4,}|\d{4,}\s*dollars', text_content)))
            
            # 10. HTML/LINK INDICATORS (in plain text)
            features['html_tags_present'] = int(bool(re.search(r'<[^>]+>', text_content)))
            features['url_count_in_text'] = len(re.findall(r'https?://[^\s]+', text_content))
            
            # 11. SUSPICIOUS PHRASES
            suspicious_phrases = [
                'verify your identity', 'confirm your identity', 
                'update your information', 'suspended account',
                'unusual activity', 'security alert', 'account locked'
            ]
            features['suspicious_phrase_count'] = sum(
                1 for phrase in suspicious_phrases if phrase in text_lower
            )
            
            # 12. URGENCY INDICATORS
            urgency_words = ['immediately', 'urgent', 'asap', 'now', 'today', 'expires']
            features['urgency_word_count'] = sum(
                text_lower.count(word) for word in urgency_words
            )
            
            # 13. TONE ANALYSIS (basic)
            features['contains_threat'] = int(
                any(word in text_lower for word in ['legal action', 'lawsuit', 'police', 'fbi'])
            )
            features['contains_reward'] = int(
                any(word in text_lower for word in ['prize', 'winner', 'won', 'lottery', 'reward'])
            )
            
        except Exception as e:
            print(f"Error extracting text features: {e}")
            # Return default values on error
            features = self._get_default_text_features()
        
        return features
    
    def extract_url_features(self, text_content):
        """
        Extract features from URLs found in email content.
        
        Parameters:
        -----------
        text_content : str
            The email content containing URLs
            
        Returns:
        --------
        dict : Dictionary containing URL-based features
        """
        features = {}
        
        try:
            # Extract all URLs from text
            urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', text_content)
            
            features['url_count'] = len(urls)
            features['has_urls'] = int(len(urls) > 0)
            features['excessive_urls'] = int(len(urls) > 5)
            
            if len(urls) > 0:
                # Analyze first URL (usually the main phishing link)
                primary_url = urls[0]
                parsed = urlparse(primary_url)
                
                # 1. DOMAIN ANALYSIS
                domain = parsed.netloc.lower()
                features['domain_length'] = len(domain)
                features['domain_has_ip'] = int(bool(re.match(r'\d+\.\d+\.\d+\.\d+', domain)))
                features['domain_has_hyphen'] = int('-' in domain)
                features['domain_has_at_symbol'] = int('@' in primary_url)
                features['domain_dots_count'] = domain.count('.')
                features['excessive_subdomains'] = int(domain.count('.') > 3)
                
                # 2. SUSPICIOUS DOMAIN PATTERNS
                features['domain_has_suspicious_tld'] = int(
                    any(domain.endswith(tld) for tld in ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz'])
                )
                features['domain_length_suspicious'] = int(len(domain) > 30)
                
                # 3. BRAND SPOOFING DETECTION
                # Check if domain contains legitimate brand names but isn't the actual domain
                brand_keywords = ['paypal', 'amazon', 'google', 'microsoft', 'apple', 
                                'facebook', 'netflix', 'bank', 'secure']
                features['possible_brand_spoofing'] = int(
                    any(brand in domain for brand in brand_keywords) and
                    not any(domain == legit for legit in self.legitimate_domains)
                )
                
                # 4. URL PATH ANALYSIS
                path = parsed.path
                features['path_length'] = len(path)
                features['path_has_suspicious_keywords'] = int(
                    any(keyword in path.lower() for keyword in 
                        ['verify', 'account', 'signin', 'login', 'update', 'confirm'])
                )
                
                # 5. URL ENCODING/OBFUSCATION
                features['url_has_encoding'] = int('%' in primary_url)
                features['url_has_redirection'] = int('//' in primary_url.split('://')[1])
                
                # 6. QUERY PARAMETERS
                query = parsed.query
                features['has_query_params'] = int(bool(query))
                features['query_param_count'] = len(query.split('&')) if query else 0
                features['suspicious_param_count'] = sum(
                    1 for param in ['user', 'id', 'token', 'session', 'redirect']
                    if param in query.lower()
                )
                
                # 7. PORT USAGE
                features['uses_non_standard_port'] = int(
                    parsed.port is not None and parsed.port not in [80, 443]
                )
                
                # 8. SHORTENED URL DETECTION
                shorteners = ['bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly']
                features['is_shortened_url'] = int(
                    any(shortener in domain for shortener in shorteners)
                )
                
                # 9. HTTPS USAGE
                features['uses_https'] = int(parsed.scheme == 'https')
                
                # 10. URL SIMILARITY TO LEGITIMATE DOMAINS
                # Check for typosquatting (e.g., paypa1.com instead of paypal.com)
                features['potential_typosquatting'] = 0
                for legit_domain in self.legitimate_domains:
                    if self._calculate_similarity(domain, legit_domain) > 0.8 and domain != legit_domain:
                        features['potential_typosquatting'] = 1
                        break
                
            else:
                # No URLs found - set defaults
                default_url_features = [
                    'domain_length', 'domain_has_ip', 'domain_has_hyphen', 
                    'domain_has_at_symbol', 'domain_dots_count', 'excessive_subdomains',
                    'domain_has_suspicious_tld', 'domain_length_suspicious',
                    'possible_brand_spoofing', 'path_length', 'path_has_suspicious_keywords',
                    'url_has_encoding', 'url_has_redirection', 'has_query_params',
                    'query_param_count', 'suspicious_param_count', 'uses_non_standard_port',
                    'is_shortened_url', 'uses_https', 'potential_typosquatting'
                ]
                for feature in default_url_features:
                    features[feature] = 0
            
        except Exception as e:
            print(f"Error extracting URL features: {e}")
            features = self._get_default_url_features()
        
        return features
    
    def extract_metadata_features(self, email_content):
        """
        Extract features from email metadata/headers.
        
        Parameters:
        -----------
        email_content : str or bytes
            Raw email content including headers
            
        Returns:
        --------
        dict : Dictionary containing metadata-based features
        """
        features = {}
        
        try:
            # Parse email
            if isinstance(email_content, str):
                email_content = email_content.encode()
            
            msg = BytesParser(policy=policy.default).parsebytes(email_content)
            
            # 1. SENDER DOMAIN ANALYSIS
            from_header = msg.get('From', '')
            sender_domain = self._extract_domain(from_header)
            features['sender_domain_length'] = len(sender_domain) if sender_domain else 0
            features['sender_has_subdomain'] = int('.' in sender_domain.split('@')[-1] if '@' in sender_domain else False)
            features['sender_has_digits'] = int(bool(re.search(r'\d', sender_domain)))
            features['sender_has_hyphen'] = int('-' in sender_domain)
            
            # 2. REPLY-TO MISMATCH
            reply_to = msg.get('Reply-To', '')
            reply_domain = self._extract_domain(reply_to)
            features['reply_to_mismatch'] = int(
                reply_domain != sender_domain and reply_to != ''
            )
            features['has_reply_to'] = int(bool(reply_to))
            
            # 3. RETURN-PATH ANALYSIS
            return_path = msg.get('Return-Path', '')
            return_domain = self._extract_domain(return_path)
            features['return_path_mismatch'] = int(
                return_domain != sender_domain and return_path != ''
            )
            
            # 4. AUTHENTICATION RECORDS
            # SPF (Sender Policy Framework)
            received_spf = msg.get('Received-SPF', '').lower()
            features['spf_pass'] = int('pass' in received_spf)
            features['spf_fail'] = int('fail' in received_spf)
            features['spf_softfail'] = int('softfail' in received_spf)
            features['spf_none'] = int(received_spf == '' or 'none' in received_spf)
            
            # DKIM (DomainKeys Identified Mail)
            auth_results = msg.get('Authentication-Results', '').lower()
            features['dkim_pass'] = int('dkim=pass' in auth_results)
            features['dkim_fail'] = int('dkim=fail' in auth_results)
            features['dkim_none'] = int('dkim=none' in auth_results or 'dkim' not in auth_results)
            
            # DMARC (Domain-based Message Authentication)
            features['dmarc_pass'] = int('dmarc=pass' in auth_results)
            features['dmarc_fail'] = int('dmarc=fail' in auth_results)
            features['dmarc_none'] = int('dmarc=none' in auth_results or 'dmarc' not in auth_results)
            
            # 5. RECEIVED HEADERS ANALYSIS
            received_headers = msg.get_all('Received', [])
            features['num_received_headers'] = len(received_headers)
            features['excessive_received_headers'] = int(len(received_headers) > 10)
            
            # Check for suspicious IPs in received headers
            suspicious_ip_count = 0
            for header in received_headers:
                # Look for private IP ranges (common in spoofing)
                if re.search(r'192\.168\.|10\.|172\.(1[6-9]|2[0-9]|3[01])\.', header):
                    suspicious_ip_count += 1
            features['suspicious_ip_count'] = suspicious_ip_count
            
            # 6. SUBJECT LINE ANALYSIS
            subject = msg.get('Subject', '')
            features['subject_length'] = len(subject)
            features['subject_has_re'] = int(subject.lower().startswith('re:'))
            features['subject_has_fwd'] = int(subject.lower().startswith('fwd:'))
            features['subject_all_caps'] = int(subject.isupper() and len(subject) > 5)
            features['subject_excessive_punctuation'] = int(
                sum(c in '!?' for c in subject) > 3
            )
            
            # Urgent/suspicious keywords in subject
            features['subject_urgent_keywords'] = sum(
                keyword in subject.lower() for keyword in self.urgent_keywords
            )
            
            # 7. MESSAGE-ID ANALYSIS
            message_id = msg.get('Message-ID', '')
            message_id_domain = self._extract_domain(message_id)
            features['message_id_domain_mismatch'] = int(
                message_id_domain != sender_domain and message_id != ''
            )
            features['has_message_id'] = int(bool(message_id))
            
            # 8. X-HEADERS ANALYSIS (often manipulated in phishing)
            x_headers = [key for key in msg.keys() if key.startswith('X-')]
            features['num_x_headers'] = len(x_headers)
            features['has_x_mailer'] = int('X-Mailer' in msg)
            features['has_x_originating_ip'] = int('X-Originating-IP' in msg)
            
            # 9. DATE/TIME ANALYSIS
            date_header = msg.get('Date', '')
            features['has_date'] = int(bool(date_header))
            
            # 10. MIME VERSION
            mime_version = msg.get('MIME-Version', '')
            features['has_mime_version'] = int(bool(mime_version))
            features['mime_version_1_0'] = int(mime_version == '1.0')
            
            # 11. CONTENT-TYPE ANALYSIS
            content_type = msg.get('Content-Type', '').lower()
            features['is_multipart'] = int('multipart' in content_type)
            features['is_html'] = int('text/html' in content_type)
            
            # 12. PRIORITY FLAGS
            priority = msg.get('X-Priority', '').lower()
            importance = msg.get('Importance', '').lower()
            features['high_priority'] = int('high' in priority or 'high' in importance or '1' in priority)
            
        except Exception as e:
            print(f"Error extracting metadata features: {e}")
            features = self._get_default_metadata_features()
            
        return features
    
    def extract_all_features(self, email_content, text_content=None):
        """
        Extract all features (text, URL, and metadata) from an email.
        
        Parameters:
        -----------
        email_content : str or bytes
            Full email content including headers (for metadata extraction)
        text_content : str, optional
            Email body text (if None, will try to extract from email_content)
            
        Returns:
        --------
        dict : Dictionary with three sub-dictionaries: 'text', 'url', 'metadata'
        """
        # If text_content not provided, try to extract it
        if text_content is None:
            try:
                if isinstance(email_content, str):
                    msg = BytesParser(policy=policy.default).parsebytes(email_content.encode())
                else:
                    msg = BytesParser(policy=policy.default).parsebytes(email_content)
                text_content = self._get_email_body(msg)
            except:
                text_content = str(email_content)
        
        all_features = {
            'text': self.extract_text_features(text_content),
            'url': self.extract_url_features(text_content),
            'metadata': self.extract_metadata_features(email_content)
        }
        
        return all_features
    
    # Helper methods
    def _extract_domain(self, email_string):
        """Extract domain from email address or header."""
        if not email_string:
            return ''
        
        # Extract email address from formatted string
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', email_string)
        if email_match:
            email_addr = email_match.group(0)
            return email_addr.split('@')[-1] if '@' in email_addr else ''
        return ''
    
    def _get_email_body(self, msg):
        """Extract plain text body from email message."""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == 'text/plain':
                    return part.get_content()
        else:
            return msg.get_content()
        return ''
    
    def _calculate_similarity(self, str1, str2):
        """Calculate simple similarity ratio between two strings."""
        # Simple Levenshtein-like similarity
        if not str1 or not str2:
            return 0.0
        
        longer = str1 if len(str1) > len(str2) else str2
        shorter = str2 if len(str1) > len(str2) else str1
        
        if len(longer) == 0:
            return 1.0
        
        # Count matching characters
        matches = sum(1 for a, b in zip(longer, shorter) if a == b)
        return matches / len(longer)
    
    def _get_default_text_features(self):
        """Return default text features dictionary."""
        return {
            'text_length': 0, 'word_count': 0, 'avg_word_length': 0,
            'sentence_count': 0, 'uppercase_ratio': 0, 'digit_ratio': 0,
            'special_char_ratio': 0, 'exclamation_count': 0, 'question_count': 0,
            'excessive_punctuation': 0, 'urgent_keyword_count': 0,
            'financial_keyword_count': 0, 'threat_keyword_count': 0,
            'has_click_here': 0, 'has_verify_account': 0, 'has_act_now': 0,
            'has_limited_time': 0, 'has_congratulations': 0, 'requests_password': 0,
            'requests_ssn': 0, 'requests_credit_card': 0, 'requests_bank_info': 0,
            'generic_greeting': 0, 'misspelling_count': 0, 'contains_currency': 0,
            'large_sum_mentioned': 0, 'html_tags_present': 0, 'url_count_in_text': 0,
            'suspicious_phrase_count': 0, 'urgency_word_count': 0,
            'contains_threat': 0, 'contains_reward': 0
        }
    
    def _get_default_url_features(self):
        """Return default URL features dictionary."""
        return {
            'url_count': 0, 'has_urls': 0, 'excessive_urls': 0,
            'domain_length': 0, 'domain_has_ip': 0, 'domain_has_hyphen': 0,
            'domain_has_at_symbol': 0, 'domain_dots_count': 0,
            'excessive_subdomains': 0, 'domain_has_suspicious_tld': 0,
            'domain_length_suspicious': 0, 'possible_brand_spoofing': 0,
            'path_length': 0, 'path_has_suspicious_keywords': 0,
            'url_has_encoding': 0, 'url_has_redirection': 0,
            'has_query_params': 0, 'query_param_count': 0,
            'suspicious_param_count': 0, 'uses_non_standard_port': 0,
            'is_shortened_url': 0, 'uses_https': 0, 'potential_typosquatting': 0
        }
    
    def _get_default_metadata_features(self):
        """Return default metadata features dictionary."""
        return {
            'sender_domain_length': 0, 'sender_has_subdomain': 0,
            'sender_has_digits': 0, 'sender_has_hyphen': 0,
            'reply_to_mismatch': 0, 'has_reply_to': 0, 'return_path_mismatch': 0,
            'spf_pass': 0, 'spf_fail': 0, 'spf_softfail': 0, 'spf_none': 0,
            'dkim_pass': 0, 'dkim_fail': 0, 'dkim_none': 0,
            'dmarc_pass': 0, 'dmarc_fail': 0, 'dmarc_none': 0,
            'num_received_headers': 0, 'excessive_received_headers': 0,
            'suspicious_ip_count': 0, 'subject_length': 0, 'subject_has_re': 0,
            'subject_has_fwd': 0, 'subject_all_caps': 0,
            'subject_excessive_punctuation': 0, 'subject_urgent_keywords': 0,
            'message_id_domain_mismatch': 0, 'has_message_id': 0,
            'num_x_headers': 0, 'has_x_mailer': 0, 'has_x_originating_ip': 0,
            'has_date': 0, 'has_mime_version': 0, 'mime_version_1_0': 0,
            'is_multipart': 0, 'is_html': 0, 'high_priority': 0
        }


# Example usage
if __name__ == "__main__":
    # Initialize extractor
    extractor = FeatureExtractor()
    
    # Example email content
    sample_email = """From: support@paypa1-secure.com
To: user@example.com
Subject: URGENT: Verify Your Account Now!
Date: Mon, 10 Apr 2025 10:00:00 +0000

Dear Valued Customer,

Your account has been temporarily suspended due to unusual activity.
Please verify your identity immediately by clicking here:
http://paypal-verify.suspicious-domain.tk/login?user=12345

If you don't act within 24 hours, your account will be permanently closed.

This is an automated message. Do not reply.
"""
    
    # Extract features
    print("Extracting text features...")
    text_features = extractor.extract_text_features(sample_email)
    print(f"Text features extracted: {len(text_features)} features")
    print(f"Sample: {list(text_features.items())[:5]}\n")
    
    print("Extracting URL features...")
    url_features = extractor.extract_url_features(sample_email)
    print(f"URL features extracted: {len(url_features)} features")
    print(f"Sample: {list(url_features.items())[:5]}\n")
    
    print("Extracting metadata features...")
    metadata_features = extractor.extract_metadata_features(sample_email)
    print(f"Metadata features extracted: {len(metadata_features)} features")
    print(f"Sample: {list(metadata_features.items())[:5]}\n")
    
    print("Extracting all features at once...")
    all_features = extractor.extract_all_features(sample_email)
    print(f"Total feature categories: {len(all_features)}")
    print(f"Text: {len(all_features['text'])} features")
    print(f"URL: {len(all_features['url'])} features")
    print(f"Metadata: {len(all_features['metadata'])} features")
