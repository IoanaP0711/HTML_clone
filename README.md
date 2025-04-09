# HTML_clone
    

A Python-based solution that groups similar HTML documents by analyzing their rendered content. The project processes a given subdirectory of HTML files, extracts the visible text or HTML structure, computes similarity between documents, and clusters them so that similar pages are grouped together.

## Overview

This project provides two alternative approaches to clustering HTML documents:

1. **TF-IDF & DBSCAN Approach:**  
   - Extracts visible text from each HTML file.
   - Converts the text into TF-IDF vectors.
   - Clusters documents using the DBSCAN algorithm based on cosine similarity.

2. **html-similarity & Agglomerative Clustering Approach:**  
   - Leverages the [html-similarity](https://github.com/matiskay/html-similarity) package to calculate structural similarity between HTML files.
   - Converts similarity scores into distances.
   - Clusters documents using Agglomerative Clustering with a precomputed distance matrix.

The output will display groups of HTML files (e.g., `[A.html, B.html], [C.html], [D.html, E.html, F.html]`) indicating which documents are similar.

## Features

- **Recursive Directory Processing:** Handles subdirectories and processes all HTML files found.
- **Flexible Similarity Measures:** Choose between a text-based (TF-IDF) approach or a structural approach using `html-similarity`.
- **Configurable Clustering:** Easily adjust clustering parameters (e.g., DBSCAN’s `eps` or the similarity threshold for Agglomerative Clustering).
- **Scalable Design:** Although designed for small datasets, the code structure allows adaptation for larger datasets.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/HTML-Clustering.git
   cd HTML_code
   python cluster_html.py clones/tier4
