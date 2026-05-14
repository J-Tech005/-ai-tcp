"""
Visualization Dashboard Helpers
Utility functions for Streamlit dashboard components
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

def plot_apfd_curves(
    ai_outcomes: List[int],
    random_outcomes: List[int],
    alphabetical_outcomes: List[int]
) -> Tuple[plt.Figure, Dict[str, float]]:
    """
    Create APFD curves comparing AI vs baselines.
    
    Args:
        ai_outcomes (List[int]): AI-ranked outcomes
        random_outcomes (List[int]): Random-ranked outcomes
        alphabetical_outcomes (List[int]): Alphabetical-ranked outcomes
    
    Returns:
        Tuple: (matplotlib figure, metrics dict)
    """
    logger.info("[Visualization] Creating APFD curves...")
    
    n = len(ai_outcomes)
    
    # Compute cumulative faults for each strategy
    ai_cumsum = np.cumsum(ai_outcomes)
    random_cumsum = np.cumsum(random_outcomes)
    alpha_cumsum = np.cumsum(alphabetical_outcomes)
    
    total_faults = ai_cumsum[-1]
    
    if total_faults == 0:
        logger.warning("[Visualization] No faults found in outcomes")
        total_faults = 1  # Avoid division by zero
    
    # Normalize to percentage
    ai_pct = (ai_cumsum / total_faults) * 100
    random_pct = (random_cumsum / total_faults) * 100
    alpha_pct = (alpha_cumsum / total_faults) * 100
    
    test_pct = (np.arange(1, n + 1) / n) * 100
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(test_pct, ai_pct, label='AI-Prioritized', linewidth=2.5, color='#2ecc71')
    ax.plot(test_pct, random_pct, label='Random', linewidth=2, color='#e74c3c', linestyle='--')
    ax.plot(test_pct, alpha_pct, label='Alphabetical', linewidth=2, color='#3498db', linestyle='--')
    
    ax.set_xlabel('% of Tests Executed', fontsize=12, fontweight='bold')
    ax.set_ylabel('% of Faults Detected', fontsize=12, fontweight='bold')
    ax.set_title('APFD Curves: Test Prioritization Strategies', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 105])
    
    logger.info("[Visualization] APFD curves created")
    return fig, {'ai': test_pct, 'random': test_pct, 'alpha': test_pct}

def plot_risk_distribution(scores: np.ndarray) -> plt.Figure:
    """
    Create histogram of risk score distribution.
    
    Args:
        scores (np.ndarray): Risk scores [0, 1]
    
    Returns:
        plt.Figure: Matplotlib figure
    """
    logger.info(f"[Visualization] Creating risk distribution plot for {len(scores)} scores")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(scores, bins=30, color='#3498db', edgecolor='black', alpha=0.7)
    ax.axvline(scores.mean(), color='#e74c3c', linestyle='--', linewidth=2, label=f'Mean: {scores.mean():.3f}')
    ax.axvline(np.median(scores), color='#f39c12', linestyle='--', linewidth=2, label=f'Median: {np.median(scores):.3f}')
    
    ax.set_xlabel('Risk Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Tests', fontsize=12, fontweight='bold')
    ax.set_title('Risk Score Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    logger.info("[Visualization] Risk distribution plot created")
    return fig

def plot_top_risks(test_ids: List[str], scores: np.ndarray, top_n: int = 20) -> plt.Figure:
    """
    Create horizontal bar chart of top N highest-risk tests.
    
    Args:
        test_ids (List[str]): Test IDs
        scores (np.ndarray): Risk scores
        top_n (int): Number of top tests to show
    
    Returns:
        plt.Figure: Matplotlib figure
    """
    logger.info(f"[Visualization] Creating top {top_n} risks plot")
    
    # Sort and get top N
    sorted_indices = np.argsort(scores)[::-1][:top_n]
    top_ids = [test_ids[i] for i in sorted_indices]
    top_scores = scores[sorted_indices]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(top_ids))
    ax.barh(y_pos, top_scores, color='#e74c3c', edgecolor='black', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_ids, fontsize=10)
    ax.set_xlabel('Risk Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Highest-Risk Tests', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    logger.info(f"[Visualization] Top {top_n} risks plot created")
    return fig

def plot_risk_threshold_pie(scores: np.ndarray, threshold: float = 0.5) -> plt.Figure:
    """
    Create pie chart of tests above vs below risk threshold.
    
    Args:
        scores (np.ndarray): Risk scores
        threshold (float): Risk threshold
    
    Returns:
        plt.Figure: Matplotlib figure
    """
    logger.info(f"[Visualization] Creating risk threshold pie chart (threshold={threshold})")
    
    above = (scores >= threshold).sum()
    below = (scores < threshold).sum()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#e74c3c', '#2ecc71']
    sizes = [above, below]
    labels = [f'High Risk (≥{threshold})\\n{above} tests', f'Low Risk (<{threshold})\\n{below} tests']
    explode = (0.05, 0)
    
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax.set_title('Risk Score Distribution by Threshold', fontsize=14, fontweight='bold')
    
    logger.info("[Visualization] Risk threshold pie chart created")
    return fig

def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]]) -> plt.Figure:
    """
    Create bar chart comparing evaluation metrics across strategies.
    
    Args:
        metrics_dict (dict): {strategy: {metric: value}}
    
    Returns:
        plt.Figure: Matplotlib figure
    """
    logger.info(f"[Visualization] Creating metrics comparison plot")
    
    df = pd.DataFrame(metrics_dict).T
    
    fig, ax = plt.subplots(figsize=(12, 6))
    df.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('Evaluation Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Strategy', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.legend(title='Metric', fontsize=10, title_fontsize=11)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=0)
    
    logger.info("[Visualization] Metrics comparison plot created")
    return fig
