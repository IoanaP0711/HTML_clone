import os
import sys
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from html_similarity import similarity

def load_html_documents(directory):
    
    documents = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.html', '.htm')):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        html_content = f.read()
                        documents.append((filepath, html_content))
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
    return documents

def compute_similarity_matrix(html_contents):
    n = len(html_contents)
    sim_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                sim = 1.0
            else:
                sim = similarity(html_contents[i], html_contents[j])
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim  
    return sim_matrix

def cluster_documents(sim_matrix, similarity_threshold=0.8):
    distance_matrix = 1 - sim_matrix
    clustering = AgglomerativeClustering(
        n_clusters=None,
        affinity='precomputed',
        linkage='average',
        distance_threshold=1 - similarity_threshold
    )
    clustering.fit(distance_matrix)
    return clustering.labels_

def group_documents_by_cluster(file_paths, cluster_labels):
    clusters = {}
    for file_path, label in zip(file_paths, cluster_labels):
        clusters.setdefault(label, []).append(file_path)
    return clusters

def main(directory):
    documents = load_html_documents(directory)
    if not documents:
        print("No HTML documents found in directory:", directory)
        return
    file_paths, html_contents = zip(*documents)
    sim_matrix = compute_similarity_matrix(html_contents)
    labels = cluster_documents(sim_matrix, similarity_threshold=0.8)
    
    clusters = group_documents_by_cluster(file_paths, labels)
    for label, files in clusters.items():
        cluster_name = f"Cluster {label}" if label != -1 else "Unique / Outlier"
        print(f"{cluster_name}:")
        for f in files:
            print(f"  - {f}")
        print()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python alternative_cluster_html.py <directory_path>")
    else:
        main(sys.argv[1])
