"""
K-Means Dashboard & Visualization Module
Generates and serves clustering charts (k selection, cluster sizes, PCA
projection) and loads summary metrics for the evaluation dashboard.
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless backend for web server environment
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

try:
    from . import config
except ImportError:
    import config


def _resolve(path, fallback_name, fallback_dir):
    if os.path.exists(path):
        return path
    alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', fallback_dir, fallback_name)
    return alt if os.path.exists(alt) else path


def load_metrics(metrics_path=None):
    """Loads k-means evaluation metrics from JSON, or returns an informative fallback."""
    path = _resolve(metrics_path or config.METRICS_JSON_PATH, 'kmeans_metrics.json', 'models')
    if not os.path.exists(path):
        return {
            'status': 'unavailable',
            'error': f"Metrics file not found at '{path}'. Please run 'src/train_and_evaluate.py'.",
        }
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data['status'] = 'available'
        return data
    except Exception as exc:
        return {'status': 'error', 'error': f"Failed to read metrics from '{path}': {exc}"}


def _load_csv(path, fallback_name):
    path = _resolve(path, fallback_name, 'reports')
    if not os.path.exists(path):
        return []
    try:
        return pd.read_csv(path).to_dict(orient='records')
    except Exception:
        return []


def load_metrics_table(table_path=None):
    return _load_csv(table_path or config.METRICS_TABLE_PATH, 'kmeans_metrics_table.csv')


def load_cluster_report(report_path=None):
    return _load_csv(report_path or config.CLUSTER_REPORT_PATH, 'kmeans_cluster_report.csv')


def generate_k_selection_chart(sweep, best_k, output_path=None):
    """Elbow (inertia) and silhouette curves over the k sweep."""
    out = output_path or config.K_SELECTION_CHART_PATH
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    ks = [r['k'] for r in sweep]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), dpi=150)
    ax1.plot(ks, [r['inertia'] for r in sweep], marker='o', color='#1F4A85')
    ax1.axvline(best_k, color='#B3261E', linestyle='--', label=f'chosen k = {best_k}')
    ax1.set_title('Elbow curve (inertia)', fontweight='bold')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Within-cluster sum of squares')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(ks, [r['silhouette'] for r in sweep], marker='o', color='#235C35')
    ax2.axvline(best_k, color='#B3261E', linestyle='--', label=f'chosen k = {best_k}')
    ax2.set_title('Silhouette score (higher is better)', fontweight='bold')
    ax2.set_xlabel('Number of clusters (k)')
    ax2.set_ylabel('Silhouette')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=170, bbox_inches='tight')
    plt.close(fig)
    return out


def generate_cluster_size_chart(profiles, output_path=None):
    """Bar chart of the number of records in every cluster."""
    out = output_path or config.CLUSTER_SIZES_CHART_PATH
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    ids = [p['cluster_id'] for p in profiles]
    sizes = [p['size'] for p in profiles]

    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=150)
    ax.bar([str(i) for i in ids], sizes, color='#1F4A85', edgecolor='#333333', linewidth=0.4)
    ax.set_title(f'K-Means cluster sizes (k = {len(ids)})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Records in cluster')
    plt.xticks(rotation=90, fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(out, dpi=170, bbox_inches='tight')
    plt.close(fig)
    return out


def generate_pca_scatter(model, X, output_path=None):
    """2-D PCA projection of the records, coloured by cluster, with centroids."""
    out = output_path or config.CLUSTER_SCATTER_CHART_PATH
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    X = np.asarray(X, dtype=float)
    pca = PCA(n_components=2, random_state=config.RANDOM_STATE).fit(X)
    pts = pca.transform(X)
    cen = pca.transform(model.cluster_centers_)

    fig, ax = plt.subplots(figsize=(11, 8), dpi=150)
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=model.labels_, cmap='nipy_spectral', s=10, alpha=0.55)
    ax.scatter(cen[:, 0], cen[:, 1], c='black', marker='X', s=90, edgecolor='white', linewidth=0.8, label='Centroids')
    ax.set_title('K-Means clusters projected to 2 principal components', fontsize=14, fontweight='bold')
    var = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f'PC1 ({var[0]:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({var[1]:.1f}% variance)')
    ax.legend(loc='best')
    ax.grid(alpha=0.25)
    fig.colorbar(sc, ax=ax, label='Cluster ID')
    plt.tight_layout()
    plt.savefig(out, dpi=170, bbox_inches='tight')
    plt.close(fig)
    return out


def verify_dashboard_assets():
    """Checks that all dashboard artifacts exist and are ready for the UI."""
    assets = {
        'k_selection_image': os.path.exists(config.K_SELECTION_CHART_PATH),
        'cluster_sizes_image': os.path.exists(config.CLUSTER_SIZES_CHART_PATH),
        'pca_scatter_image': os.path.exists(config.CLUSTER_SCATTER_CHART_PATH),
        'metrics_json': os.path.exists(config.METRICS_JSON_PATH),
        'metrics_table_csv': os.path.exists(config.METRICS_TABLE_PATH),
        'cluster_report_csv': os.path.exists(config.CLUSTER_REPORT_PATH),
        'trained_model': os.path.exists(config.KMEANS_MODEL_PATH),
    }
    missing = [k for k, v in assets.items() if not v]
    return {
        'all_present': not missing,
        'assets': assets,
        'missing': missing,
        'metrics_data': load_metrics() if assets['metrics_json'] else None,
    }


if __name__ == '__main__':
    print(json.dumps(verify_dashboard_assets(), indent=2))
