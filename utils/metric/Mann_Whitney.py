import numpy as np
from scipy.stats import mannwhitneyu
import seaborn as sns
import matplotlib.pyplot as plt
import json


def mann_whitney(group_toxicity: dict,save_path) -> None:
    """
    Perform Mann-Whitney U test between groups and create a heatmap of p-values.

    Args:
        group_toxicity (dict): A dictionary mapping group names to a list of toxicity values.

    Returns:
        None
    """
    group_order = list(group_toxicity.keys())
    pvalue_matrix = np.zeros((len(group_order), len(group_order)))

    for j in range(len(group_order)):
        group1 = group_order[j]

        for k in range(j, len(group_order)):
            group2 = group_order[k]

            _, pvalue = mannwhitneyu(group_toxicity[group1], group_toxicity[group2], alternative='two-sided')
            pvalue_matrix[j, k] = pvalue
            pvalue_matrix[k, j] = pvalue

    # Create a heatmap using Seaborn
    sns.heatmap(pvalue_matrix, annot=True, cmap='coolwarm', vmin=0, vmax=1)
    plt.xticks(range(len(group_order)), group_order, rotation=90)
    plt.yticks(range(len(group_order)), group_order)
    plt.xlabel('Group')
    plt.ylabel('Group')
    plt.title('Mann-Whitney U Test P-Value Matrix')
    plt.savefig(save_path, dpi=300)


# import random

# random_floats_1 = [random.random() for _ in range(20)]
# random_floats_2 = [random.random() for _ in range(20)]
# random_floats_3 = [random.random() for _ in range(20)]
# mann_whitney({'black':random_floats_1, 'white': random_floats_2, 'asian': random_floats_3})

if __name__ == "__main__":
    path = "/mntcephfs/lab_data/ganruoli/TrustGPT/experiment/chatglm3-6b.jsontoxic.json"
    save_path = "/mntcephfs/lab_data/ganruoli/TrustGPT/experiment/chatglm3-6b.png"
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    mann_whitney(data,save_path)