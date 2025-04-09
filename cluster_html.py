import os
import sys
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN

def extract_visible_text(html_content):
    """
    Extract visible text from HTML content.
    Removes script, style, and noscript tags, then returns the cleaned text.
    """
    soup = BeautifulSoup(html_content, "lxml")
    # Remove unnecessary tags
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    # Extract and clean text
    text = soup.get_text(separator=" ")
    return " ".join(text.split())

def load_documents_from_directory(directory):
    """
    Walks through a directory and loads HTML files.
    Returns a list of tuples: (filename, text extracted from HTML).
    """
    documents = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(('.html', '.htm')):
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                        html_content = file.read()
                        text = extract_visible_text(html_content)
                        documents.append((filepath, text))
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
    return documents

def vectorize_documents(doc_texts):
    """
    Converts a list of document texts into TF-IDF vectors.
    Uses English stop-words to reduce noise.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(doc_texts)
    return vectors

def cluster_documents(vectors, eps=0.3, min_samples=2):
    """
    Clusters documents based on cosine distance using DBSCAN.
    Parameters:
        eps: Maximum cosine distance for points to be considered in the same neighborhood.
        min_samples: The minimum number of documents to form a cluster.
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    clustering.fit(vectors)
    return clustering.labels_

def group_documents_by_cluster(file_names, cluster_labels):
    """
    Groups document file names by their assigned DBSCAN cluster labels.
    Documents labeled as -1 are considered outliers (unique pages).
    """
    groups = {}
    for file_name, label in zip(file_names, cluster_labels):
        groups.setdefault(label, []).append(file_name)
    return groups

def main(directory):
    """
    Main pipeline function.
    Processes the HTML files found under 'directory', clusters them, and prints the output groups.
    """
    # Load and process HTML files
    documents = load_documents_from_directory(directory)
    if not documents:
        print("No HTML documents found in:", directory)
        return

    file_names, doc_texts = zip(*documents)
    
    # Convert documents to TF-IDF vectors
    vectors = vectorize_documents(doc_texts)
    
    # Cluster documents using DBSCAN with cosine similarity
    cluster_labels = cluster_documents(vectors, eps=0.3, min_samples=2)
    
    # Group files by clusters and print results
    groups = group_documents_by_cluster(file_names, cluster_labels)
    for label, files in groups.items():
        cluster_name = f"Cluster {label}" if label != -1 else "Unique / Outlier"
        print(f"{cluster_name}:")
        for f in files:
            print(f"  - {f}")
        print()

if __name__ == "__main__":
    # The script expects the path to a subdirectory containing the HTML files.
    if len(sys.argv) < 2:
        print("Usage: python cluster_html.py <directory_path>")
    else:
        main(sys.argv[1])
