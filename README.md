# K-Means Clustering on NYC Taxi Trip Data

## Project Overview

This project implements the K-Means clustering algorithm from scratch in Python. It analyzes taxi trip data to group trips into meaningful segments based on distance, fare, and tip patterns.

**What the project does:**
- Loads taxi trip data (real CSV or synthetic fallback)
- Finds natural groupings of trips using K-Means clustering
- Determines the optimal number of clusters using the elbow method
- Visualizes results with clear graphs
- Compares custom implementation against scikit-learn

**Business value:** Understanding trip segments helps taxi companies optimize driver allocation, predict demand, and personalize services.

---

## Dataset

**Source:** NYC Taxi Trip Records (AWS Open Data Registry) or synthetic fallback

| Feature | Description | Unit |
|---------|-------------|------|
| `trip_distance` | Length of trip | Miles |
| `fare_amount` | Total fare paid | US Dollars |
| `tip_amount` | Tip left by passenger | US Dollars |
| `passenger_count` | Number of riders | Count |

**Data loading priority:**
1. Looks for `data/taxi_sample.csv` if you provide it
2. If not found, automatically generates realistic synthetic data

You don't need to download anything. The code runs immediately.

---

## Tools Used

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.9+ | Core programming language |
| NumPy | 1.24.3 | Mathematical operations, distance calculations |
| Pandas | 2.0.3 | Data loading and manipulation |
| scikit-learn | 1.3.0 | Baseline comparison and standardization |
| Matplotlib | 3.7.1 | Visualizations (elbow curve, cluster plots) |
| Seaborn | 0.12.2 | Enhanced visualizations |
| Pytest | 7.4.0 | Unit testing |

**Development Environment:** VS Code (PyCharm also works)

---

## Installation
### Step 1: Check Python is installed
```bash
python3 --version
```
### Step 2: Create project folder and navigate in
```
mkdir kmeans_clustering_project
cd kmeans_clustering_project
```
### Step 3: Create folder structure
```
mkdir src tests output
```
### Step 4: Create empty __init__.py files (important for Python imports)
```
touch src/__init__.py
touch tests/__init__.py
```
### Step 5: Create requirements.txt
```
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.1
seaborn==0.12.2
pytest==7.4.0
```
### Step 6: Install dependencies

# Try this first
```
pip3 install -r requirements.txt
```

# If that fails, try this
```
python3 -m pip install -r requirements.txt
```
**Note for macOS users:** Use `pip3` not `pip`

### How to Run
Run the main application
```
python3 run_clustering.py
```
Run unit tests only
```
python3 -m pytest tests/ -v
```
Skip tests and run clustering only
```
python3 run_clustering.py --skip-tests
```

## Book:

```
 Géron, A. 2022. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow. 3rd ed. O'Reilly Media: Sebastopol, CA.
```
## Article:

```
Arthur, D. & Vassilvitskii, S. 2007. K-means++: The Advantages of Careful Seeding. Proceedings of the Eighteenth Annual ACM-SIAM Symposium on Discrete Algorithms. 1027-1035.
```

## Website:

```
Amazon Web Services. 2024. Registry of Open Data on AWS [WWW Document]. AWS. URL https://registry.opendata.aws/ [Accessed 20 April 2026].
```

## Code reference:

```
Pedregosa, F. et al. 2011. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research. 12:2825-2830.
```

## Project Structure
```
clustering/
│
├── src/
│   ├── __init__.py          # Empty file (tells Python this is a package)
│   └── kmeans_clustering.py # Main K-Means implementation
│
├── tests/
│   ├── __init__.py          # Empty file (tells Python this is a package)
│   └── test_kmeans.py       # Unit tests
│
├── data/                    # Place your taxi_sample.csv here (optional)
├── output/                  # Generated images appear here
├── requirements.txt         # Python dependencies
├── run_clustering.py        # Main execution script
└── README.md                # This file
```