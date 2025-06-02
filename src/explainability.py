import shap
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

def explain_scores(resume_embeddings, labels):
    """
    Generate SHAP explanations for a Logistic Regression model trained on resume embeddings.

    Args:
        resume_embeddings (list or np.array): List or array of embedding vectors (shape: n_samples x n_features).
        labels (list or np.array): Real similarity scores or binary labels (e.g., 1 for good match, 0 for bad).

    Returns:
        explainer: SHAP explainer object.
        shap_values: SHAP values array.
        model: trained LogisticRegression model.
    """
    X = np.array(resume_embeddings)
    y = np.array(labels)

    # Train logistic regression model
    model = LogisticRegression(max_iter=1000).fit(X, y)

    # Create SHAP explainer using the trained model and the dataset
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    return explainer, shap_values, model

def plot_shap_summary(shap_values, feature_names=None):
    """
    Plot a SHAP summary plot for the explanations.

    Args:
        shap_values: SHAP values object returned by explainer.
        feature_names (list, optional): Names of features (embedding dims).
    """
    shap.summary_plot(shap_values.values, features=shap_values.data, feature_names=feature_names)
