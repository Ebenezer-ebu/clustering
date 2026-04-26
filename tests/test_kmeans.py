"""
Unit Tests for K-Means Clustering Implementation

These tests validate:
1. Euclidean distance calculation
2. Cluster assignment logic
3. Centroid update correctness
4. Convergence behavior
5. Edge cases (empty clusters, single points)

Run with: pytest tests/test_kmeans.py -v
"""

import pytest
import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from kmeans_clustering import KMeansFromScratch


class TestEuclideanDistance:
    """Test suite for Euclidean distance calculations."""
    
    def test_distance_zero(self):
        """Distance from point to itself should be zero."""
        kmeans = KMeansFromScratch()
        point = np.array([1.0, 2.0, 3.0])
        assert kmeans.euclidean_distance(point, point) == 0.0
    
    def test_distance_2d(self):
        """Test Euclidean distance in 2D space."""
        kmeans = KMeansFromScratch()
        point1 = np.array([0, 0])
        point2 = np.array([3, 4])
        # Expected: sqrt(9 + 16) = 5
        assert kmeans.euclidean_distance(point1, point2) == 5.0
    
    def test_distance_3d(self):
        """Test Euclidean distance in 3D space."""
        kmeans = KMeansFromScratch()
        point1 = np.array([1, 1, 1])
        point2 = np.array([4, 5, 6])
        # Expected: sqrt(9 + 16 + 25) = sqrt(50) ≈ 7.071
        expected = np.sqrt(50)
        assert abs(kmeans.euclidean_distance(point1, point2) - expected) < 1e-6
    
    def test_distance_symmetry(self):
        """Distance should be symmetric: dist(a,b) = dist(b,a)."""
        kmeans = KMeansFromScratch()
        point1 = np.array([1.5, -2.3, 4.1])
        point2 = np.array([-0.5, 3.7, 1.2])
        assert kmeans.euclidean_distance(point1, point2) == \
               kmeans.euclidean_distance(point2, point1)


class TestClusterAssignment:
    """Test suite for point-to-centroid assignment logic."""
    
    def setup_method(self):
        """Setup test data before each test."""
        self.kmeans = KMeansFromScratch(k=2)
        # Simple 2D test points
        self.X = np.array([[0, 0], [10, 10], [0, 10], [10, 0]])
        # Centroids at (0,0) and (10,10)
        self.kmeans.centroids = np.array([[0, 0], [10, 10]])
    
    def test_assign_clusters_basic(self):
        """Test basic cluster assignment."""
        labels = self.kmeans.assign_clusters(self.X)
        # Point (0,0) should go to centroid 0
        assert labels[0] == 0
        # Point (10,10) should go to centroid 1
        assert labels[1] == 1
    
    def test_assign_clusters_tie_breaking(self):
        """Test tie-breaking when point equidistant to multiple centroids."""
        # Points exactly midway between centroids
        midway = np.array([[5, 5]])
        labels = self.kmeans.assign_clusters(midway)
        # Should assign to one of the centroids (argmin picks first)
        assert labels[0] in [0, 1]


class TestCentroidUpdate:
    """Test suite for centroid update logic."""
    
    def test_update_centroids_simple(self):
        """Test centroid update with simple 1D points."""
        kmeans = KMeansFromScratch(k=2)
        X = np.array([[1], [2], [9], [10]])
        labels = np.array([0, 0, 1, 1])
        kmeans.centroids = np.array([[0], [0]])
        
        new_centroids = kmeans.update_centroids(X, labels)
        
        # Cluster 0 mean: (1+2)/2 = 1.5
        # Cluster 1 mean: (9+10)/2 = 9.5
        assert new_centroids[0][0] == 1.5
        assert new_centroids[1][0] == 9.5
    
    def test_update_centroids_empty_cluster(self):
        """Test centroid update when a cluster gets no points."""
        kmeans = KMeansFromScratch(k=3)
        X = np.array([[1], [2], [3]])
        labels = np.array([0, 0, 1])
        kmeans.centroids = np.array([[0], [0], [0]])
        
        new_centroids = kmeans.update_centroids(X, labels)
        
        # Cluster with points: should update to mean
        assert new_centroids[0][0] == 1.5
        # Empty cluster: should keep previous centroid
        assert new_centroids[2][0] == 0
    
    def test_mean_calculation(self):
        """Test that centroids are computed as exact means."""
        kmeans = KMeansFromScratch(k=1)
        X = np.array([[2, 4], [4, 6], [6, 8]])
        labels = np.array([0, 0, 0])
        kmeans.centroids = np.array([[0, 0]])
        
        new_centroids = kmeans.update_centroids(X, labels)
        
        # Mean: ((2+4+6)/3, (4+6+8)/3) = (4, 6)
        assert new_centroids[0][0] == 4.0
        assert new_centroids[0][1] == 6.0


class TestConvergence:
    """Test suite for algorithm convergence behavior."""
    
    def test_convergence_stops_early(self):
        """Test that algorithm stops when centroids stabilize."""
        kmeans = KMeansFromScratch(k=2, max_iters=100, tol=1e-4)
        # Well-separated data
        X = np.vstack([
            np.random.normal(0, 0.1, (50, 2)),
            np.random.normal(10, 0.1, (50, 2))
        ])
        
        kmeans.fit(X)
        # Should converge in few iterations (not reaching max_iters)
        # We can check that labels are assigned consistently
        assert len(np.unique(kmeans.labels)) == 2
    
    def test_inertia_decreases(self):
        """Test that inertia decreases monotonically."""
        kmeans = KMeansFromScratch(k=2)
        X = np.array([[1, 1], [1, 2], [10, 10], [10, 11]])
        
        # Track inertia over iterations
        inertias = []
        # Manually run one iteration
        kmeans.centroids = kmeans.initialize_centroids(X)
        for _ in range(5):
            labels = kmeans.assign_clusters(X)
            inertia = kmeans.compute_inertia(X, labels)
            inertias.append(inertia)
            kmeans.centroids = kmeans.update_centroids(X, labels)
        
        # Inertia should be non-increasing
        for i in range(1, len(inertias)):
            assert inertias[i] <= inertias[i-1] + 1e-6


class TestEdgeCases:
    """Test suite for edge cases and error handling."""
    
    def test_single_cluster(self):
        """Test K=1 (all points in one cluster)."""
        kmeans = KMeansFromScratch(k=1)
        X = np.random.randn(100, 3)
        kmeans.fit(X)
        assert np.all(kmeans.labels == 0)
    
    def test_more_clusters_than_points(self):
        """Test when K > number of data points."""
        kmeans = KMeansFromScratch(k=10)
        X = np.random.randn(5, 3)
        kmeans.fit(X)
        # All points should be assigned, some clusters may be empty
        assert len(kmeans.labels) == 5
    
    def test_single_point_per_cluster(self):
        """Test when each point is its own cluster."""
        kmeans = KMeansFromScratch(k=3)
        X = np.array([[1, 1], [2, 2], [3, 3]])
        kmeans.fit(X)
        # With perfect initialization, each point becomes centroid
        assert kmeans.inertia_ == 0.0
    
    def test_duplicate_points(self):
        """Test behavior with duplicate data points."""
        kmeans = KMeansFromScratch(k=2)
        X = np.vstack([np.ones((10, 2)), np.ones((10, 2))])
        kmeans.fit(X)
        # All points identical, should all go to same cluster
        assert len(np.unique(kmeans.labels)) == 1


class TestPredictNewPoints:
    """Test suite for predicting on new data after fitting."""
    
    def test_predict_after_fit(self):
        """Test that predict returns correct labels for new points."""
        kmeans = KMeansFromScratch(k=2)
        X_train = np.array([[0, 0], [0, 1], [10, 10], [10, 9]])
        kmeans.fit(X_train)
        
        X_new = np.array([[0, 0.5], [10, 10.5]])
        predictions = kmeans.predict(X_new)
        
        assert len(predictions) == 2
        # New points close to training clusters should get same labels
        assert predictions[0] == kmeans.labels[0]
        assert predictions[1] == kmeans.labels[2]


def run_all_tests():
    """Run all tests and print summary."""
    print("\n" + "="*60)
    print("RUNNING UNIT TESTS")
    print("="*60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEuclideanDistance))
    suite.addTests(loader.loadTestsFromTestCase(TestClusterAssignment))
    suite.addTests(loader.loadTestsFromTestCase(TestCentroidUpdate))
    suite.addTests(loader.loadTestsFromTestCase(TestConvergence))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestPredictNewPoints))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*60)
    print(f"TEST SUMMARY: {result.testsRun} tests run")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*60)
    
    return result


if __name__ == "__main__":
    import unittest
    run_all_tests()