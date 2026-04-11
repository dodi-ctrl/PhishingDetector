"""
Metadata Agent for Phishing Detection
Uses Random Forest Classifier to analyze email metadata features.

Author: Ifanyi Uche Henry
Project: Social Engineering Attacks (Phishing) Detection using NLP
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json


class MetadataAgent:
    """
    Random Forest-based agent for analyzing email metadata.
    Detects phishing attempts through header analysis, authentication records,
    and sender information validation.
    """
    
    def __init__(self, n_estimators=100, max_depth=20, random_state=42):
        """
        Initialize the Metadata Agent.
        
        Parameters:
        -----------
        n_estimators : int
            Number of trees in the random forest
        max_depth : int
            Maximum depth of the trees
        random_state : int
            Random seed for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            class_weight='balanced',  # Handle class imbalance
            n_jobs=-1  # Use all available cores
        )
        
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        self.training_history = {}
        self.best_params = None
        
    def prepare_features(self, metadata_features_list):
        """
        Convert list of metadata feature dictionaries to DataFrame.
        
        Parameters:
        -----------
        metadata_features_list : list of dict
            List of metadata feature dictionaries from FeatureExtractor
            
        Returns:
        --------
        pd.DataFrame : Feature matrix ready for training/prediction
        """
        df = pd.DataFrame(metadata_features_list)
        
        # Store feature names
        self.feature_names = df.columns.tolist()
        
        # Handle any missing values
        df = df.fillna(0)
        
        return df
    
    def train(self, X_train, y_train, validate=True, tune_hyperparameters=False):
        """
        Train the Random Forest model on metadata features.
        
        Parameters:
        -----------
        X_train : pd.DataFrame or array-like
            Training features (metadata dictionaries converted to DataFrame)
        y_train : array-like
            Training labels (0=legitimate, 1=phishing)
        validate : bool
            Whether to perform cross-validation
        tune_hyperparameters : bool
            Whether to perform hyperparameter tuning using GridSearchCV
            
        Returns:
        --------
        self : MetadataAgent
            Trained agent instance
        """
        print("\n" + "="*70)
        print("TRAINING METADATA AGENT (Random Forest)")
        print("="*70)
        
        # Convert to DataFrame if necessary
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train_array = X_train.values
        else:
            X_train_array = np.array(X_train)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_array)
        
        # Hyperparameter tuning (optional but recommended)
        if tune_hyperparameters:
            print("\nPerforming hyperparameter tuning...")
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            grid_search = GridSearchCV(
                self.model, param_grid, cv=5, 
                scoring='f1', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train_scaled, y_train)
            
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            print(f"\nBest parameters: {self.best_params}")
            print(f"Best F1 score: {grid_search.best_score_:.4f}")
        else:
            # Train with default parameters
            self.model.fit(X_train_scaled, y_train)
        
        self.is_trained = True
        
        # Cross-validation
        if validate:
            print("\nPerforming 5-fold cross-validation...")
            cv_scores = cross_val_score(
                self.model, X_train_scaled, y_train, 
                cv=5, scoring='accuracy'
            )
            cv_f1_scores = cross_val_score(
                self.model, X_train_scaled, y_train, 
                cv=5, scoring='f1'
            )
            
            print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"Cross-validation F1-Score: {cv_f1_scores.mean():.4f} (+/- {cv_f1_scores.std() * 2:.4f})")
            
            # Store CV results
            self.training_history['cv_accuracy'] = cv_scores
            self.training_history['cv_f1'] = cv_f1_scores
        
        # Feature importance analysis
        self._analyze_feature_importance()
        
        # Store training metadata
        self.training_history['training_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.training_history['num_samples'] = len(X_train_array)
        self.training_history['num_features'] = len(self.feature_names)
        
        print("\n✓ Metadata Agent training completed successfully!")
        
        return self
    
    def predict(self, X_test):
        """
        Predict labels for test data.
        
        Parameters:
        -----------
        X_test : pd.DataFrame or array-like
            Test features (metadata dictionaries)
            
        Returns:
        --------
        array : Predicted labels (0=legitimate, 1=phishing)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions!")
        
        # Convert to array if necessary
        if isinstance(X_test, pd.DataFrame):
            X_test_array = X_test.values
        else:
            X_test_array = np.array(X_test)
        
        # Scale features
        X_test_scaled = self.scaler.transform(X_test_array)
        
        return self.model.predict(X_test_scaled)
    
    def predict_proba(self, X_test):
        """
        Predict probability estimates for test data.
        
        Parameters:
        -----------
        X_test : pd.DataFrame or array-like
            Test features
            
        Returns:
        --------
        array : Probability estimates [prob_legitimate, prob_phishing]
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions!")
        
        # Convert to array if necessary
        if isinstance(X_test, pd.DataFrame):
            X_test_array = X_test.values
        else:
            X_test_array = np.array(X_test)
        
        # Scale features
        X_test_scaled = self.scaler.transform(X_test_array)
        
        return self.model.predict_proba(X_test_scaled)
    
    def evaluate(self, X_test, y_test, plot_results=True):
        """
        Comprehensive evaluation of model performance.
        
        Parameters:
        -----------
        X_test : pd.DataFrame or array-like
            Test features
        y_test : array-like
            True labels
        plot_results : bool
            Whether to generate visualization plots
            
        Returns:
        --------
        dict : Dictionary containing all evaluation metrics
        """
        print("\n" + "="*70)
        print("METADATA AGENT PERFORMANCE EVALUATION")
        print("="*70)
        
        # Make predictions
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]  # Phishing probability
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = conf_matrix.ravel()
        
        # Additional metrics
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
        
        # ROC-AUC
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except:
            roc_auc = None
        
        # Compile results
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'false_positive_rate': fpr,
            'false_negative_rate': fnr,
            'true_negative_rate': tnr,
            'roc_auc': roc_auc
        }
        
        # Print results
        print(f"\nAccuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
        if roc_auc:
            print(f"ROC-AUC:   {roc_auc:.4f}")
        
        print(f"\nFalse Positive Rate (FPR): {fpr:.4f} ({fpr*100:.2f}%)")
        print(f"False Negative Rate (FNR): {fnr:.4f} ({fnr*100:.2f}%)")
        print(f"True Negative Rate (TNR):  {tnr:.4f} ({tnr*100:.2f}%)")
        
        print("\n" + "-"*70)
        print("CONFUSION MATRIX")
        print("-"*70)
        print(f"{'':15} {'Predicted Legit':>15} {'Predicted Phishing':>18}")
        print(f"{'Actual Legit':15} {tn:>15} {fp:>18}")
        print(f"{'Actual Phishing':15} {fn:>15} {tp:>18}")
        
        print("\n" + "-"*70)
        print("DETAILED CLASSIFICATION REPORT")
        print("-"*70)
        print(classification_report(y_test, y_pred, 
                                   target_names=['Legitimate', 'Phishing'],
                                   digits=4))
        
        # Generate plots
        if plot_results:
            self._plot_evaluation_results(conf_matrix, y_test, y_pred_proba, results)
        
        return results
    
    def _analyze_feature_importance(self):
        """Analyze and display feature importance from Random Forest."""
        if not self.is_trained:
            return
        
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n" + "-"*70)
        print("TOP 15 MOST IMPORTANT METADATA FEATURES")
        print("-"*70)
        
        for idx, row in feature_importance_df.head(15).iterrows():
            print(f"{row['feature']:35} {row['importance']:.6f}")
        
        # Store for later use
        self.training_history['feature_importance'] = feature_importance_df.to_dict('records')
        
        return feature_importance_df
    
    def _plot_evaluation_results(self, conf_matrix, y_test, y_pred_proba, results):
        """Generate visualization plots for model evaluation."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Metadata Agent Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix Heatmap
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Legitimate', 'Phishing'],
                   yticklabels=['Legitimate', 'Phishing'],
                   ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_ylabel('Actual')
        axes[0, 0].set_xlabel('Predicted')
        
        # 2. Feature Importance (Top 10)
        feature_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        axes[0, 1].barh(range(len(feature_imp)), feature_imp['importance'])
        axes[0, 1].set_yticks(range(len(feature_imp)))
        axes[0, 1].set_yticklabels(feature_imp['feature'])
        axes[0, 1].set_xlabel('Importance')
        axes[0, 1].set_title('Top 10 Most Important Features')
        axes[0, 1].invert_yaxis()
        
        # 3. ROC Curve
        if results['roc_auc'] is not None:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            axes[1, 0].plot(fpr, tpr, label=f'ROC (AUC = {results["roc_auc"]:.4f})', 
                          linewidth=2)
            axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            axes[1, 0].set_xlabel('False Positive Rate')
            axes[1, 0].set_ylabel('True Positive Rate')
            axes[1, 0].set_title('ROC Curve')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Performance Metrics Bar Chart
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metrics_values = [results['accuracy'], results['precision'], 
                         results['recall'], results['f1_score']]
        
        bars = axes[1, 1].bar(metrics_names, metrics_values, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
        axes[1, 1].set_ylim([0, 1.1])
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.4f}',
                          ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('metadata_agent_evaluation.png', dpi=300, bbox_inches='tight')
        print("\n✓ Evaluation plots saved as 'metadata_agent_evaluation.png'")
        plt.show()
    
    def get_prediction_with_confidence(self, metadata_features):
        """
        Get prediction with confidence score for a single email.
        
        Parameters:
        -----------
        metadata_features : dict
            Single metadata feature dictionary
            
        Returns:
        --------
        dict : Prediction result with confidence and verdict
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions!")
        
        # Convert to DataFrame
        features_df = pd.DataFrame([metadata_features])
        
        # Ensure all features are present
        for feature in self.feature_names:
            if feature not in features_df.columns:
                features_df[feature] = 0
        
        # Reorder columns
        features_df = features_df[self.feature_names]
        
        # Get prediction and probability
        prediction = self.predict(features_df)[0]
        probabilities = self.predict_proba(features_df)[0]
        
        result = {
            'verdict': 'Phishing' if prediction == 1 else 'Legitimate',
            'confidence': float(probabilities[1] if prediction == 1 else probabilities[0]),
            'phishing_probability': float(probabilities[1]),
            'legitimate_probability': float(probabilities[0])
        }
        
        return result
    
    def save_model(self, filepath='metadata_agent.pkl'):
        """
        Save the trained model and associated data to disk.
        
        Parameters:
        -----------
        filepath : str
            Path where the model will be saved
        """
        if not self.is_trained:
            raise ValueError("Cannot save an untrained model!")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained,
            'training_history': self.training_history,
            'best_params': self.best_params
        }
        
        joblib.dump(model_data, filepath)
        print(f"\n✓ Model saved successfully to: {filepath}")
    
    def load_model(self, filepath='metadata_agent.pkl'):
        """
        Load a trained model from disk.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model file
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        self.training_history = model_data.get('training_history', {})
        self.best_params = model_data.get('best_params', None)
        
        print(f"\n✓ Model loaded successfully from: {filepath}")
        print(f"  Features: {len(self.feature_names)}")
        if self.training_history:
            print(f"  Training date: {self.training_history.get('training_date', 'Unknown')}")
            print(f"  Training samples: {self.training_history.get('num_samples', 'Unknown')}")
    
    def export_results(self, results, filepath='metadata_agent_results.json'):
        """
        Export evaluation results to JSON file.
        
        Parameters:
        -----------
        results : dict
            Results dictionary from evaluate() method
        filepath : str
            Path to save the JSON file
        """
        # Convert numpy types to Python types for JSON serialization
        results_clean = {}
        for key, value in results.items():
            if isinstance(value, (np.integer, np.floating)):
                results_clean[key] = float(value)
            else:
                results_clean[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(results_clean, f, indent=4)
        
        print(f"\n✓ Results exported to: {filepath}")


# Example usage and testing
if __name__ == "__main__":
    print("Metadata Agent - Random Forest Classifier")
    print("="*70)
    
    # Example: Create sample data
    print("\nGenerating sample metadata features for demonstration...")
    
    # Simulated metadata features for 100 emails (50 phishing, 50 legitimate)
    np.random.seed(42)
    
    # Legitimate emails (label = 0)
    legit_features = []
    for _ in range(50):
        features = {
            'sender_domain_length': np.random.randint(10, 20),
            'sender_has_subdomain': np.random.choice([0, 1], p=[0.7, 0.3]),
            'sender_has_digits': np.random.choice([0, 1], p=[0.9, 0.1]),
            'spf_pass': 1,
            'dkim_pass': 1,
            'dmarc_pass': 1,
            'reply_to_mismatch': 0,
            'num_received_headers': np.random.randint(3, 8),
            'subject_urgent_keywords': 0,
            'high_priority': np.random.choice([0, 1], p=[0.9, 0.1])
        }
        legit_features.append(features)
    
    # Phishing emails (label = 1)
    phishing_features = []
    for _ in range(50):
        features = {
            'sender_domain_length': np.random.randint(25, 40),
            'sender_has_subdomain': np.random.choice([0, 1], p=[0.3, 0.7]),
            'sender_has_digits': np.random.choice([0, 1], p=[0.3, 0.7]),
            'spf_pass': 0,
            'dkim_pass': 0,
            'dmarc_pass': 0,
            'reply_to_mismatch': 1,
            'num_received_headers': np.random.randint(8, 15),
            'subject_urgent_keywords': np.random.randint(1, 4),
            'high_priority': np.random.choice([0, 1], p=[0.3, 0.7])
        }
        phishing_features.append(features)
    
    # Combine and create labels
    all_features = legit_features + phishing_features
    labels = [0] * 50 + [1] * 50
    
    # Initialize agent
    agent = MetadataAgent(n_estimators=100, random_state=42)
    
    # Prepare features
    X = agent.prepare_features(all_features)
    y = np.array(labels)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train the agent
    agent.train(X_train, y_train, validate=True, tune_hyperparameters=False)
    
    # Evaluate
    results = agent.evaluate(X_test, y_test, plot_results=False)
    
    # Test single prediction
    print("\n" + "="*70)
    print("TESTING SINGLE EMAIL PREDICTION")
    print("="*70)
    
    test_email_metadata = {
        'sender_domain_length': 35,
        'sender_has_subdomain': 1,
        'sender_has_digits': 1,
        'spf_pass': 0,
        'dkim_pass': 0,
        'dmarc_pass': 0,
        'reply_to_mismatch': 1,
        'num_received_headers': 12,
        'subject_urgent_keywords': 3,
        'high_priority': 1
    }
    
    prediction = agent.get_prediction_with_confidence(test_email_metadata)
    print(f"\nVerdict: {prediction['verdict']}")
    print(f"Confidence: {prediction['confidence']:.4f}")
    print(f"Phishing Probability: {prediction['phishing_probability']:.4f}")
    
    # Save model
    agent.save_model('metadata_agent_example.pkl')
    
    print("\n" + "="*70)
    print("✓ Metadata Agent demonstration complete!")
    print("="*70)
