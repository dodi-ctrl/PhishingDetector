"""
URL Agent for Phishing Detection
Uses Random Forest Classifier to analyse URL-based features extracted from emails.
Author: Udodinachukwu David Nwoke
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


class URLAgent:
    """
    Random Forest-based agent for analysing URL features in emails.
    Detects phishing attempts through domain analysis, path inspection,
    query parameter examination, and link structure evaluation.

    URL features examined:
        - Presence and count of URLs
        - Domain properties (IP address use, hyphens, length, subdomains, TLD)
        - Brand spoofing and typosquatting indicators
        - Path and query string suspicious patterns
        - URL encoding, redirection, and non-standard ports
        - HTTPS usage and URL shortener detection
    """

    def __init__(self, n_estimators=100, max_depth=20, random_state=42):
        """
        Initialise the URL Agent.

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
            n_jobs=-1                 # Use all available cores
        )

        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        self.training_history = {}
        self.best_params = None

    def prepare_features(self, url_features_list):
        """
        Convert a list of URL feature dictionaries to a DataFrame.

        Parameters:
        -----------
        url_features_list : list of dict
            List of URL feature dictionaries from FeatureExtractor

        Returns:
        --------
        pd.DataFrame : Feature matrix ready for training/prediction
        """
        df = pd.DataFrame(url_features_list)
        self.feature_names = df.columns.tolist()
        df = df.fillna(0)
        return df

    def train(self, X_train, y_train, validate=True, tune_hyperparameters=False):
        """
        Train the Random Forest model on URL features.

        Parameters:
        -----------
        X_train : pd.DataFrame or array-like
            Training features
        y_train : array-like
            Training labels (0=legitimate, 1=phishing)
        validate : bool
            Whether to perform cross-validation
        tune_hyperparameters : bool
            Whether to perform hyperparameter tuning using GridSearchCV

        Returns:
        --------
        self : URLAgent
            Trained agent instance
        """
        print("\n" + "="*70)
        print("TRAINING URL AGENT (Random Forest)")
        print("="*70)

        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train_array = X_train.values
        else:
            X_train_array = np.array(X_train)

        X_train_scaled = self.scaler.fit_transform(X_train_array)

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
            self.model.fit(X_train_scaled, y_train)

        self.is_trained = True

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

            self.training_history['cv_accuracy'] = cv_scores
            self.training_history['cv_f1'] = cv_f1_scores

        self._analyze_feature_importance()

        self.training_history['training_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.training_history['num_samples'] = len(X_train_array)
        self.training_history['num_features'] = len(self.feature_names)

        print("\n✓ URL Agent training completed successfully!")
        return self

    def predict(self, X_test):
        """
        Predict labels for test data.

        Parameters:
        -----------
        X_test : pd.DataFrame or array-like
            Test features

        Returns:
        --------
        array : Predicted labels (0=legitimate, 1=phishing)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions!")

        if isinstance(X_test, pd.DataFrame):
            X_test_array = X_test.values
        else:
            X_test_array = np.array(X_test)

        return self.model.predict(self.scaler.transform(X_test_array))

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

        if isinstance(X_test, pd.DataFrame):
            X_test_array = X_test.values
        else:
            X_test_array = np.array(X_test)

        return self.model.predict_proba(self.scaler.transform(X_test_array))

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
            Whether to generate visualisation plots

        Returns:
        --------
        dict : Dictionary containing all evaluation metrics
        """
        print("\n" + "="*70)
        print("URL AGENT PERFORMANCE EVALUATION")
        print("="*70)

        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]

        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall    = recall_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred)

        conf_matrix = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = conf_matrix.ravel()

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0

        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except Exception:
            roc_auc = None

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

        print(f"\nAccuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
        if roc_auc is not None:
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

        if plot_results:
            self._plot_evaluation_results(conf_matrix, y_test, y_pred_proba, results)

        return results

    def _analyze_feature_importance(self):
        """Analyse and display feature importance from the Random Forest."""
        if not self.is_trained:
            return

        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\n" + "-"*70)
        print("TOP URL FEATURES BY IMPORTANCE")
        print("-"*70)

        for _, row in feature_importance_df.head(15).iterrows():
            print(f"{row['feature']:35} {row['importance']:.6f}")

        self.training_history['feature_importance'] = feature_importance_df.to_dict('records')
        return feature_importance_df

    def _plot_evaluation_results(self, conf_matrix, y_test, y_pred_proba, results):
        """Generate visualisation plots for model evaluation."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('URL Agent Performance Analysis', fontsize=16, fontweight='bold')

        # 1. Confusion Matrix Heatmap
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Legitimate', 'Phishing'],
                    yticklabels=['Legitimate', 'Phishing'],
                    ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_ylabel('Actual')
        axes[0, 0].set_xlabel('Predicted')

        # 2. Feature Importance (all URL features — there are only 23)
        feature_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        axes[0, 1].barh(range(len(feature_imp)), feature_imp['importance'])
        axes[0, 1].set_yticks(range(len(feature_imp)))
        axes[0, 1].set_yticklabels(feature_imp['feature'], fontsize=8)
        axes[0, 1].set_xlabel('Importance')
        axes[0, 1].set_title('URL Feature Importance (all features)')
        axes[0, 1].invert_yaxis()

        # 3. ROC Curve
        if results['roc_auc'] is not None:
            fpr_vals, tpr_vals, _ = roc_curve(y_test, y_pred_proba)
            axes[1, 0].plot(fpr_vals, tpr_vals,
                            label=f'ROC (AUC = {results["roc_auc"]:.4f})',
                            linewidth=2)
            axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            axes[1, 0].set_xlabel('False Positive Rate')
            axes[1, 0].set_ylabel('True Positive Rate')
            axes[1, 0].set_title('ROC Curve')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # 4. Performance Metrics Bar Chart
        metrics_names  = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metrics_values = [results['accuracy'], results['precision'],
                          results['recall'], results['f1_score']]

        bars = axes[1, 1].bar(metrics_names, metrics_values,
                               color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
        axes[1, 1].set_ylim([0, 1.1])
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2., height,
                            f'{height:.4f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig('url_agent_evaluation.png', dpi=300, bbox_inches='tight')
        print("\n✓ Evaluation plots saved as 'url_agent_evaluation.png'")
        plt.show()

    def get_prediction_with_confidence(self, url_features):
        """
        Get prediction with confidence score for a single email.

        Parameters:
        -----------
        url_features : dict
            Single URL feature dictionary from FeatureExtractor.extract_url_features()

        Returns:
        --------
        dict : Prediction result with verdict, confidence, and probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions!")

        features_df = pd.DataFrame([url_features])

        # Pad any features the model expects that are missing from this input
        for feature in self.feature_names:
            if feature not in features_df.columns:
                features_df[feature] = 0

        features_df = features_df[self.feature_names]

        prediction    = self.predict(features_df)[0]
        probabilities = self.predict_proba(features_df)[0]

        return {
            'verdict': 'Phishing' if prediction == 1 else 'Legitimate',
            'confidence': float(probabilities[1] if prediction == 1 else probabilities[0]),
            'phishing_probability': float(probabilities[1]),
            'legitimate_probability': float(probabilities[0])
        }

    def save_model(self, filepath='url_agent.pkl'):
        """
        Save the trained model and associated data to disk.

        Parameters:
        -----------
        filepath : str
            Path where the model will be saved
        """
        if not self.is_trained:
            raise ValueError("Cannot save an untrained model!")

        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained,
            'training_history': self.training_history,
            'best_params': self.best_params
        }, filepath)

        print(f"\n✓ Model saved successfully to: {filepath}")

    def load_model(self, filepath='url_agent.pkl'):
        """
        Load a trained model from disk.

        Parameters:
        -----------
        filepath : str
            Path to the saved model file
        """
        model_data = joblib.load(filepath)

        self.model            = model_data['model']
        self.scaler           = model_data['scaler']
        self.feature_names    = model_data['feature_names']
        self.is_trained       = model_data['is_trained']
        self.training_history = model_data.get('training_history', {})
        self.best_params      = model_data.get('best_params', None)

        print(f"\n✓ Model loaded successfully from: {filepath}")
        print(f"  Features: {len(self.feature_names)}")
        if self.training_history:
            print(f"  Training date:    {self.training_history.get('training_date', 'Unknown')}")
            print(f"  Training samples: {self.training_history.get('num_samples', 'Unknown')}")

    def export_results(self, results, filepath='url_agent_results.json'):
        """
        Export evaluation results to a JSON file.

        Parameters:
        -----------
        results : dict
            Results dictionary from evaluate()
        filepath : str
            Path to save the JSON file
        """
        results_clean = {
            key: float(val) if isinstance(val, (np.integer, np.floating)) else val
            for key, val in results.items()
        }

        with open(filepath, 'w') as f:
            json.dump(results_clean, f, indent=4)

        print(f"\n✓ Results exported to: {filepath}")


if __name__ == "__main__":
    from dataset_handling import (
        build_eml_corpus,
        build_url_corpus,
        extract_url_features_from_url_corpus,
    )
    from feature_extraction import FeatureExtractor

    print("="*70)
    print("URL AGENT — Training on Multiple Corpora")
    print("(phishing_pot + Nazario phishing  |  Enron ham)")
    print("="*70)

    # ----------------------------------------
    # STEP 1: Configure data paths
    # Same three sources as the Metadata Agent — keeps training data
    # consistent across all three agents.
    # Update after downloading in Colab:
    #   !git clone https://github.com/rf-peixoto/phishing_pot.git
    #   !wget https://monkey.org/~jose/phishing/phishing3.mbox
    # ----------------------------------------
    PHISHING_DIRS  = ['phishing_pot/emails']
    PHISHING_MBOX  = ['phishing3.mbox']
    ENRON_MAX      = 5000

    # ----------------------------------------
    # STEP 2: Build EML corpus (email-embedded URLs)
    # ----------------------------------------
    eml_df = build_eml_corpus(
        phishing_dirs=PHISHING_DIRS,
        phishing_mbox_paths=PHISHING_MBOX,
        enron_max=ENRON_MAX,
        nazario_after_year=2022,
    )

    # ----------------------------------------
    # STEP 3: Build URL corpus from .eml bodies
    # ----------------------------------------
    url_df = build_url_corpus(
        eml_df=eml_df if len(eml_df) > 0 else None,
    )

    if len(url_df) == 0:
        print("\n✗ No URLs loaded. Check data paths and re-run.")
        exit(1)

    # ----------------------------------------
    # STEP 4: Extract URL features
    # ----------------------------------------
    extractor = FeatureExtractor()
    features_df, labels = extract_url_features_from_url_corpus(url_df, extractor)

    # ----------------------------------------
    # STEP 5: Train / test split
    # ----------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, labels,
        test_size=0.3,
        random_state=42,
        stratify=labels,
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples:     {len(X_test)}")
    print(f"  - Legitimate:   {(y_test == 0).sum()}")
    print(f"  - Phishing:     {(y_test == 1).sum()}")

    # ----------------------------------------
    # STEP 6: Train the agent
    # ----------------------------------------
    agent = URLAgent(n_estimators=100, max_depth=20, random_state=42)
    agent.train(X_train, y_train, validate=True, tune_hyperparameters=False)

    # ----------------------------------------
    # STEP 7: Evaluate
    # ----------------------------------------
    results = agent.evaluate(X_test, y_test, plot_results=True)

    # ----------------------------------------
    # STEP 8: Save model and results
    # ----------------------------------------
    agent.save_model('url_agent.pkl')
    agent.export_results(results, 'url_agent_results.json')

    print("\n" + "="*70)
    print("✓ URL Agent training complete!")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  F1-Score:  {results['f1_score']:.4f}")
    print(f"  ROC-AUC:   {results['roc_auc']:.4f}")
    print("="*70)
