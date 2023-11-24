from scipy.linalg import eigh
from sklearn.cluster import KMeans
import numpy as np
import networkx as nx

def norm_laplacian(A):
    # Calculate the degree matrix
    D = np.diag(np.sum(A, axis=1))
    
    # Calculate D^(-1/2)
    D_sqrt_inv = np.linalg.inv(np.sqrt(D))
    
    # Calculate the normalized Laplacian matrix
    L = np.dot(D_sqrt_inv, np.dot(D - A, D_sqrt_inv))
    
    return L

def spectral_embedding(L, n_components):
    """
    Compute Laplacian Eigenmaps.

    Parameters:
        L (np.array): Normalized Laplacian matrix.
        n_components (int): Number of eigenvectors (embedding dimensions) to compute.

    Returns:
        np.array: Laplacian Eigenmaps with n_components columns.
    """
    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigh(L, eigvals=(1, n_components))
    
    # Sort the eigenvectors by eigenvalues
    sorted_indices = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Return the Laplacian Eigenmaps
    return eigenvectors


def spectral_clustering(G, n_clusters, n_components):
    A = nx.to_numpy_array(G)
    L = norm_laplacian(A)
    embedding = spectral_embedding(L, n_components)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(embedding)
    return kmeans.labels_, embedding