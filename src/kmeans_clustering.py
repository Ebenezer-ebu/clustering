"""
K-Means Clustering Implementation

This module implements the K-Means clustering algorithm from scratch
and using scikit-learn for comparison. The implementation follows
standard K-Means algorithm steps:
1. Initialize K centroids randomly
2. Assign each data point to nearest centroid (Euclidean distance)
3. Update centroids as mean of assigned points
4. Repeat steps 2-3 until convergence

Dataset: NYC Taxi Trip Records or similar numerical dataset
Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class KMeansFromScratch:
    """
    K-Means clustering algorithm implemented from scratch.
    
    The algorithm finds K clusters by iteratively:
    - Assigning points to nearest centroid
    - Recalculating centroids as cluster means
    
    Attributes:
        k (int): Number of clusters
        max_iters (int): Maximum iterations for convergence
        tol (float): Tolerance for convergence (centroid movement threshold)
        centroids (np.ndarray): Final centroid locations
        labels (np.ndarray): Cluster assignments for each data point
        inertia_ (float): Sum of squared distances to nearest centroid
    """
    
    def __init__(self, k: int = 3, max_iters: int = 100, tol: float = 1e-4):
        """
        Initialize K-Means clustering algorithm.
        
        Args:
            k: Number of clusters to form
            max_iters: Maximum number of iterations
            tol: Tolerance for convergence - if centroids move less than tol,
                 algorithm stops early
                 
        Example:
            >>> kmeans = KMeansFromScratch(k=3, max_iters=100)
        """
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels = None
        self.inertia_ = None
        
    def euclidean_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Euclidean distance formula: sqrt(sum((x_i - y_i)^2))
        
        Args:
            point1: First point as numpy array
            point2: Second point as numpy array
            
        Returns:
            float: Euclidean distance between points
            
        Example:
            >>> kmeans = KMeansFromScratch()
            >>> dist = kmeans.euclidean_distance(np.array([0,0]), np.array([3,4]))
            >>> print(dist)  # Output: 5.0
        """
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    def initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """Initialize centroids using k-means++ method."""
        n_samples, n_features = X.shape
        centroids = np.zeros((self.k, n_features), dtype=float)
        
        # First centroid: random point
        first_idx = np.random.choice(n_samples)
        centroids[0] = X[first_idx]
        
        # Remaining centroids
        for i in range(1, self.k):
            # Calculate squared distances to nearest existing centroid
            distances = np.zeros(n_samples)
            for j in range(n_samples):
                min_dist = np.inf
                for c in range(i):
                    dist = self.euclidean_distance(X[j], centroids[c])
                    min_dist = min(min_dist, dist)
                distances[j] = min_dist ** 2
            
            # FIX: If all distances are zero, choose randomly
            if np.sum(distances) == 0:
                remaining_indices = [idx for idx in range(n_samples) 
                                    if not any(np.array_equal(X[idx], centroids[c]) 
                                            for c in range(i))]
                if remaining_indices:
                    next_idx = np.random.choice(remaining_indices)
                else:
                    next_idx = np.random.choice(n_samples)
            else:
                probabilities = distances / np.sum(distances)
                next_idx = np.random.choice(n_samples, p=probabilities)
            
            centroids[i] = X[next_idx]
        
        return centroids
    
    def assign_clusters(self, X: np.ndarray) -> np.ndarray:
        """
        Assign each data point to the nearest centroid.
        
        For each point, calculate distance to all centroids and select
        the centroid with minimum distance.
        
        Args:
            X: Input data array of shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Cluster assignments as integer labels (0 to k-1)
        """
        n_samples = X.shape[0]
        labels = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Calculate distances to all centroids
            distances = np.zeros(self.k)
            for j in range(self.k):
                distances[j] = self.euclidean_distance(X[i], self.centroids[j])
            
            # Assign to closest centroid
            labels[i] = np.argmin(distances)
            
        return labels.astype(int)
    
    def update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Update centroids as mean of points assigned to each cluster."""
        new_centroids = np.zeros_like(self.centroids, dtype=float)  # Force float type
        
        for j in range(self.k):
            cluster_points = X[labels == j]
            if len(cluster_points) > 0:
                new_centroids[j] = np.mean(cluster_points, axis=0)
            else:
                new_centroids[j] = self.centroids[j]
        
        return new_centroids
    
    def compute_inertia(self, X: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute inertia (sum of squared distances to nearest centroid).
        
        Lower inertia indicates tighter, more cohesive clusters.
        Used as the objective function that K-Means minimizes.
        
        Args:
            X: Input data array
            labels: Cluster assignments
            
        Returns:
            float: Sum of squared distances from points to their centroids
        """
        inertia = 0.0
        for j in range(self.k):
            cluster_points = X[labels == j]
            if len(cluster_points) > 0:
                # Sum squared distances from points to cluster centroid
                distances = np.linalg.norm(cluster_points - self.centroids[j], axis=1)
                inertia += np.sum(distances ** 2)
        return inertia
    
    def elbow_method(self, X: np.ndarray, max_k: int = 10) -> Tuple[List[int], List[float]]:
        """
        Determine optimal number of clusters using elbow method.
        
        The elbow method plots inertia vs. number of clusters.
        The optimal k is at the "elbow" where diminishing returns begin.
        
        Args:
            X: Input data array
            max_k: Maximum number of clusters to test
            
        Returns:
            Tuple of (k_values, inertia_values) for plotting
        """
        k_values = []
        inertia_values = []
        
        for k in range(1, max_k + 1):
            kmeans = KMeansFromScratch(k=k, max_iters=100)
            kmeans.fit(X)
            k_values.append(k)
            inertia_values.append(kmeans.inertia_)
            print(f"K={k}: Inertia={kmeans.inertia_:.2f}")
            
        return k_values, inertia_values
    
    def fit(self, X: np.ndarray) -> 'KMeansFromScratch':
        """
        Run the K-Means clustering algorithm.
        
        Main training loop:
        1. Initialize centroids
        2. Repeat until convergence or max iterations:
           a) Assign points to nearest centroids
           b) Update centroids to cluster means
           c) Check if centroids moved less than tolerance
        
        Args:
            X: Input data array of shape (n_samples, n_features)
            
        Returns:
            self: Fitted KMeansFromScratch instance
            
        Example:
            >>> kmeans = KMeansFromScratch(k=3)
            >>> kmeans.fit(data)
            >>> print(kmeans.labels)
        """
        n_samples, n_features = X.shape
        
        # Initialize centroids
        self.centroids = self.initialize_centroids(X)
        
        for iteration in range(self.max_iters):
            # Assign points to clusters
            new_labels = self.assign_clusters(X)
            
            # Update centroids
            new_centroids = self.update_centroids(X, new_labels)
            
            # Check for convergence
            centroid_shift = np.sum(np.abs(new_centroids - self.centroids))
            
            self.centroids = new_centroids
            self.labels = new_labels
            
            if centroid_shift < self.tol:
                print(f"Converged after {iteration + 1} iterations")
                break
                
        # Compute final inertia
        self.inertia_ = self.compute_inertia(X, self.labels)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data points.
        
        Args:
            X: New data points of shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Cluster assignments for each new point
        """
        return self.assign_clusters(X)


class DataLoader:
    """Handle data loading, preprocessing, and validation."""
    
    @staticmethod
    def load_taxi_sample(filepath: str) -> pd.DataFrame:
        """
        Load NYC taxi sample data.
        
        Expected columns (from NYC taxi dataset):
        - trip_distance: Distance of trip in miles
        - fare_amount: Fare paid
        - tip_amount: Tip amount
        - passenger_count: Number of passengers
        - pickup_longitude, pickup_latitude: Pickup coordinates
        - dropoff_longitude, dropoff_latitude: Dropoff coordinates
        
        If file not found, generates synthetic taxi-like data for demonstration.
        """
        try:
            df = pd.read_csv(filepath)
            print(f"Loaded {len(df)} records from {filepath}")
            return df
        except FileNotFoundError:
            print(f"File {filepath} not found. Generating synthetic taxi data...")
            return DataLoader.generate_synthetic_taxi_data()
    
    @staticmethod
    def generate_synthetic_taxi_data(n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic taxi trip data for demonstration.
        
        This mimics the structure of NYC taxi data with realistic
        distributions for:
        - Trip distance (0.5 to 15 miles, right-skewed)
        - Fare amount ($3 to $50)
        - Tip amount (0% to 30% of fare)
        - Passenger count (1 to 6)
        """
        np.random.seed(42)
        
        # Generate realistic distributions
        trip_distance = np.random.exponential(scale=2.5, size=n_samples)
        trip_distance = np.clip(trip_distance, 0.5, 15.0)
        
        # Fare based on distance plus base rate
        fare_amount = 2.5 + (trip_distance * 1.5) + np.random.normal(0, 2, n_samples)
        fare_amount = np.clip(fare_amount, 3.0, 50.0)
        
        # Tip as percentage of fare (0% to 30% with some zeros)
        tip_percentage = np.random.choice([0, 0.1, 0.15, 0.2, 0.25, 0.3], 
                                          size=n_samples, 
                                          p=[0.3, 0.1, 0.2, 0.2, 0.1, 0.1])
        tip_amount = fare_amount * tip_percentage
        
        passenger_count = np.random.choice([1, 2, 3, 4, 5, 6], size=n_samples, 
                                          p=[0.5, 0.2, 0.15, 0.08, 0.05, 0.02])
        
        df = pd.DataFrame({
            'trip_distance': trip_distance,
            'fare_amount': fare_amount,
            'tip_amount': tip_amount,
            'passenger_count': passenger_count
        })
        
        return df
    
    @staticmethod
    def prepare_features(df: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
        """
        Prepare and normalize features for clustering.
        
        Steps:
        1. Select specified feature columns
        2. Handle missing values (fill with median)
        3. Standardize features (zero mean, unit variance)
        
        Standardization is critical for K-Means because it uses Euclidean
        distance. Features with larger scales would otherwise dominate.
        """
        # Select features
        X = df[feature_columns].copy()
        
        # Handle missing values
        if X.isnull().any().any():
            print("Handling missing values with median imputation")
            X = X.fillna(X.median())
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print(f"Prepared {X_scaled.shape[0]} samples with {X_scaled.shape[1]} features")
        print(f"Feature means: {scaler.mean_}")
        print(f"Feature scales: {scaler.scale_}")
        
        return X_scaled, scaler


class ClusterVisualizer:
    """Create visualizations for clustering results."""
    
    @staticmethod
    def plot_elbow_curve(k_values: List[int], inertia_values: List[float], 
                         save_path: Optional[str] = None):
        """
        Plot elbow curve for determining optimal K.
        
        The optimal K is at the "elbow" where the rate of decrease
        in inertia slows dramatically.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, inertia_values, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Number of Clusters (K)', fontsize=12)
        plt.ylabel('Inertia (Sum of Squared Distances)', fontsize=12)
        plt.title('Elbow Method for Optimal K Selection', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Elbow plot saved to {save_path}")
        plt.show()
    
    @staticmethod
    def plot_clusters_2d(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray,
                         feature_names: List[str], save_path: Optional[str] = None):
        """
        Create 2D scatter plot of clusters using first two features.
        
        Points are colored by cluster assignment, with centroids marked as stars.
        """
        plt.figure(figsize=(12, 8))
        
        # Scatter plot by cluster
        scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', 
                             s=50, alpha=0.7, edgecolors='w', linewidth=0.5)
        
        # Mark centroids
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='*', 
                   s=300, edgecolors='black', linewidth=2, label='Centroids')
        
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel(feature_names[0], fontsize=12)
        plt.ylabel(feature_names[1], fontsize=12)
        plt.title('K-Means Clustering Results', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Cluster plot saved to {save_path}")
        plt.show()
    
    @staticmethod
    def plot_feature_distributions(df: pd.DataFrame, feature_columns: List[str],
                                   save_path: Optional[str] = None):
        """
        Plot feature distributions for exploratory data analysis.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, col in enumerate(feature_columns):
            if idx < len(axes):
                axes[idx].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
                axes[idx].set_title(f'Distribution of {col}', fontsize=12)
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frequency')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Feature distribution plot saved to {save_path}")
        plt.show()


def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> Dict:
    """
    Evaluate clustering quality using standard metrics.
    
    Metrics:
    - Silhouette Score: Measures how similar points are to their own cluster
      compared to other clusters. Range [-1, 1]. Higher is better.
      
    - Calinski-Harabasz Index: Ratio of between-cluster variance to within-cluster
      variance. Higher indicates better defined clusters.
    
    Returns:
        Dictionary containing evaluation metrics
    """
    if len(set(labels)) < 2:
        return {"error": "Need at least 2 clusters for evaluation"}
    
    try:
        silhouette = silhouette_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)
        
        print("\n" + "="*50)
        print("CLUSTERING EVALUATION METRICS")
        print("="*50)
        print(f"Silhouette Score: {silhouette:.4f}")
        print(f"  - Range: -1 to 1")
        print(f"  - Interpretation: >0.5 = good clustering, >0.7 = strong clustering")
        print(f"\nCalinski-Harabasz Index: {calinski:.2f}")
        print(f"  - Higher values indicate better defined clusters")
        print("="*50)
        
        return {
            "silhouette_score": silhouette,
            "calinski_harabasz_score": calinski
        }
    except Exception as e:
        print(f"Evaluation error: {e}")
        return {"error": str(e)}


def main():
    """
    Main execution function for K-Means clustering pipeline.
    
    Pipeline steps:
    1. Load and explore data
    2. Preprocess features
    3. Find optimal K using elbow method
    4. Run K-Means clustering
    5. Evaluate and visualize results
    """
    print("\n" + "="*60)
    print("K-MEANS CLUSTERING APPLICATION")
    print("Dataset: NYC Taxi Trip Records (or synthetic equivalent)")
    print("="*60)
    
    # Step 1: Load data
    print("\n[Step 1] Loading data...")
    data_loader = DataLoader()
    df = data_loader.load_taxi_sample('data/taxi_sample.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Step 2: Define features for clustering
    feature_columns = ['trip_distance', 'fare_amount', 'tip_amount']
    
    # Step 3: Prepare and normalize features
    print("\n[Step 2] Preparing features...")
    X_scaled, scaler = data_loader.prepare_features(df, feature_columns)
    
    # Step 4: Visualize feature distributions (optional)
    visualizer = ClusterVisualizer()
    visualizer.plot_feature_distributions(df, feature_columns, 'output/feature_distributions.png')
    
    # Step 5: Find optimal K using elbow method
    print("\n[Step 3] Finding optimal number of clusters...")
    kmeans_optimizer = KMeansFromScratch(k=3)  # Temporary instance
    k_values, inertia_values = kmeans_optimizer.elbow_method(X_scaled, max_k=10)
    visualizer.plot_elbow_curve(k_values, inertia_values, 'output/elbow_curve.png')
    
    # Step 6: Run K-Means with selected K
    optimal_k = 3  # Based on elbow curve
    print(f"\n[Step 4] Running K-Means with K={optimal_k}...")
    kmeans = KMeansFromScratch(k=optimal_k, max_iters=100)
    kmeans.fit(X_scaled)
    
    # Step 7: Evaluate results
    print("\n[Step 5] Evaluating clustering quality...")
    metrics = evaluate_clustering(X_scaled, kmeans.labels)
    
    # Step 8: Visualize clusters
    print("\n[Step 6] Generating visualizations...")
    visualizer.plot_clusters_2d(X_scaled, kmeans.labels, kmeans.centroids,
                               feature_names=feature_columns, 
                               save_path='output/cluster_plot.png')
    
    # Step 9: Compare with scikit-learn implementation
    print("\n[Step 7] Comparing with scikit-learn implementation...")
    sklearn_kmeans = SklearnKMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    sklearn_labels = sklearn_kmeans.fit_predict(X_scaled)
    sklearn_metrics = evaluate_clustering(X_scaled, sklearn_labels)
    
    # Step 10: Summary output
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"Number of samples clustered: {len(X_scaled)}")
    print(f"Number of features used: {X_scaled.shape[1]}")
    print(f"Number of clusters (K): {optimal_k}")
    print(f"Custom implementation inertia: {kmeans.inertia_:.2f}")
    print(f"Scikit-learn implementation inertia: {sklearn_kmeans.inertia_:.2f}")
    
    print("\nCluster size distribution:")
    unique, counts = np.unique(kmeans.labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        percentage = (count / len(kmeans.labels)) * 100
        print(f"  Cluster {cluster}: {count} points ({percentage:.1f}%)")
    
    print("\n" + "="*60)
    print("Clustering complete!")
    print("="*60)
    
    return kmeans, X_scaled, metrics


if __name__ == "__main__":
    kmeans_model, clustered_data, evaluation_metrics = main()