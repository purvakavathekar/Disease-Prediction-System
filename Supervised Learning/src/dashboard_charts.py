"""
Medical Triage Dashboard & Visualization Module
Generates and serves evaluation charts, confusion matrix visualization,
feature importance rankings, and summary performance metrics.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless backend for web server environment
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

try:
    from . import config
except ImportError:
    import config

def load_metrics(metrics_path=None):
    """
    Safely loads triage evaluation metrics from JSON file.
    Returns structured dictionary with actual metrics or informative error fallback.
    """
    path = metrics_path or config.METRICS_JSON_PATH
    if not os.path.exists(path):
        alt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'triage_metrics.json')
        if os.path.exists(alt_path):
            path = alt_path

    if not os.path.exists(path):
        return {
            'status': 'unavailable',
            'error': f"Metrics file not found at '{path}'. Please run 'src/train_and_evaluate.py'.",
            'accuracy': None,
            'macro_f1': None,
            'emergency_recall': None,
            'under_triage_rate': None,
            'over_triage_rate': None
        }

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data['status'] = 'available'
            return data
    except Exception as exc:
        return {
            'status': 'error',
            'error': f"Failed to read metrics from '{path}': {str(exc)}",
            'accuracy': None,
            'macro_f1': None
        }

def load_metrics_table(table_path=None):
    """Safely loads triage metrics summary table from CSV."""
    path = table_path or config.METRICS_TABLE_PATH
    if not os.path.exists(path):
        alt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'reports', 'triage_metrics_table.csv')
        if os.path.exists(alt_path):
            path = alt_path

    if not os.path.exists(path):
        return []

    try:
        df = pd.read_csv(path)
        return df.to_dict(orient='records')
    except Exception:
        return []

def load_classification_report(report_path=None):
    """Safely loads per-class classification report from CSV."""
    path = report_path or config.CLASSIFICATION_REPORT_PATH
    if not os.path.exists(path):
        alt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'reports', 'triage_classification_report.csv')
        if os.path.exists(alt_path):
            path = alt_path

    if not os.path.exists(path):
        return []

    try:
        df = pd.read_csv(path)
        return df.to_dict(orient='records')
    except Exception:
        return []

def generate_confusion_matrix_chart(y_true, y_pred, labels, output_path=None):
    """
    Generates a readable, high-resolution Confusion Matrix chart with actual class labels.
    Saves to output_path (default config.CONFUSION_MATRIX_PATH).
    """
    out_file = output_path or config.CONFUSION_MATRIX_PATH
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Format class labels cleanly (strip excessive whitespace, truncate if needed)
    display_labels = [str(lbl).strip() for lbl in labels]

    fig, ax = plt.subplots(figsize=(18, 16), dpi=150)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=display_labels,
        yticklabels=display_labels,
        cbar_kws={'label': 'Sample Count'},
        linewidths=0.5,
        linecolor='#e0e0e0',
        ax=ax
    )

    ax.set_title(
        f"Medical Triage Multi-Class Confusion Matrix (N={len(y_true)} Held-Out Test Samples, 41 Disease Categories)",
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    ax.set_xlabel("Predicted Diagnosis / Triage Label", fontsize=13, fontweight='semibold', labelpad=12)
    ax.set_ylabel("True Diagnosis / Triage Label", fontsize=13, fontweight='semibold', labelpad=12)
    plt.xticks(rotation=90, fontsize=9, ha='center')
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    plt.savefig(out_file, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return out_file

def generate_feature_importance_chart(model, feature_names, output_path=None, top_n=20):
    """
    Generates a feature importance ranking bar chart for models that support it
    (e.g., RandomForestClassifier, GradientBoosting).
    If model has no feature_importances_, generates an informative explanatory chart.
    """
    out_file = output_path or config.FEATURE_IMPORTANCE_PATH
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i].replace('_', ' ').title() for i in indices]
        top_scores = importances[indices]

        # Horizontal bar chart with gradient palette
        colors = sns.color_palette("mako", len(top_scores))[::-1]
        bars = ax.barh(range(len(top_scores)), top_scores[::-1], color=colors, edgecolor='#333333', linewidth=0.6)
        ax.set_yticks(range(len(top_scores)))
        ax.set_yticklabels(top_features[::-1], fontsize=10)
        ax.set_xlabel("Gini Feature Importance Score", fontsize=12, fontweight='semibold', labelpad=10)
        ax.set_title(
            f"Top {top_n} Most Influential Symptom Features in Medical Triage Model",
            fontsize=14,
            fontweight='bold',
            pad=15
        )

        # Annotate bar values
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + (max(top_scores) * 0.01),
                bar.get_y() + bar.get_height() / 2,
                f"{width:.4f}",
                va='center',
                ha='left',
                fontsize=9,
                color='#333333'
            )
        ax.grid(axis='x', linestyle='--', alpha=0.5)
    else:
        # Graceful fallback for models without feature importances (e.g., KNN)
        model_name = type(model).__name__
        ax.text(
            0.5, 0.5,
            f"Feature Importance Not Supported by {model_name}\n\n"
            f"Distance-based classifiers like {model_name} use geometric neighbor consensus\n"
            f"rather than tree splits or feature weighting.\n"
            f"Feature importance is available for tree-based ensemble models (e.g. Random Forest).",
            ha='center', va='center', fontsize=12, color='#555555',
            bbox=dict(boxstyle="round,pad=1.5", facecolor="#f8f9fa", edgecolor="#dee2e6")
        )
        ax.set_title(f"Model Feature Importance ({model_name})", fontsize=14, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_file, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return out_file

def verify_dashboard_assets():
    """
    Verifies that all required dashboard metrics and visualization assets exist,
    are readable, and are ready for UI rendering.
    """
    assets = {
        'confusion_matrix_image': os.path.exists(config.CONFUSION_MATRIX_PATH),
        'feature_importance_image': os.path.exists(config.FEATURE_IMPORTANCE_PATH),
        'metrics_json': os.path.exists(config.METRICS_JSON_PATH),
        'metrics_table_csv': os.path.exists(config.METRICS_TABLE_PATH),
        'classification_report_csv': os.path.exists(config.CLASSIFICATION_REPORT_PATH),
        'trained_model': os.path.exists(config.MODEL_JOB_PATH) or os.path.exists(config.LEGACY_MODEL_PATH)
    }

    all_present = all(assets.values())
    missing = [k for k, v in assets.items() if not v]

    return {
        'all_present': all_present,
        'assets': assets,
        'missing': missing,
        'metrics_data': load_metrics() if assets['metrics_json'] else None
    }

if __name__ == '__main__':
    print("Verifying Medical Triage Dashboard Assets...")
    status = verify_dashboard_assets()
    print(json.dumps(status, indent=2))
