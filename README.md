# COMP 472 Project (Fall 2026) · Team 18

This repository contains our code for the COMP 472 Kaggle project. The project has two classification tasks:

* **Image:** 500 features per image, 50 classes, 20,000 training examples.
* **Sentiment:** 768 BERT features per text, 3 classes (0 = Negative, 1 = Neutral, 2 = Positive), 12,000 training examples.

We compare k-Nearest Neighbours, a Decision Tree, a Random Forest and a feed-forward Neural Network on both tasks. We also run a k-Means clustering experiment. k-NN, the Decision Tree, the Random Forest and k-Means are written from scratch with NumPy. The Neural Network uses PyTorch. scikit-learn is used only for the train/validation split and for metrics.

## Files

| File | Purpose | Files it writes |
|:--|:--|:--|
| `KNN.ipynb` | k-NN from scratch. Experiments on k, feature standardization, cosine distance and distance-weighted voting. | `knn_experiments.csv`, `knn_results.csv`, `submission_knn.csv` |
| `Decision_Tree.ipynb` | Decision Tree from scratch. Experiments on maximum depth, minimum samples per leaf, minimum samples to split, and Gini vs. entropy. | `submission_decision_tree.csv` |
| `Random_Forest.ipynb` | Random Forest from scratch (bootstrap samples, random features at each split, majority vote). Experiments on depth, features per split and number of trees. Comparison with the single Decision Tree, and error analysis. | `random_forest_results.csv`, `submission_random_forest.csv` |
| `Neural_Network.ipynb` | Feed-forward network in PyTorch. Experiments on learning rate, hidden layers, activation, dropout, weight decay and batch size. Error analysis and comparison with k-NN. **Writes our final Kaggle submission.** | `submission_knn_image_nn_sentiment.csv` |
| `Clustering.ipynb` | k-Means from scratch on the sentiment data. Purity, ARI and NMI against the true labels, for k = 1 to 10. Not part of the Kaggle submission. | none |
| `README.md` | This file. | |

## Setup

1. Install Python 3.12 or newer.
2. Install the libraries:

   ```bash
   pip install numpy pandas matplotlib scikit-learn torch jupyter
   ```

   A GPU is not needed. Everything runs on a CPU.

3. Download the five CSV files from the Kaggle competition page (Data tab) and put them **in the same folder as the notebooks**:

   ```
   image_train.csv   image_test.csv   sentiment_train.csv   sentiment_test.csv   sample_submission.csv
   ```

   The data files are not included in this repository.

## How to run

Open a notebook in Jupyter and run all cells from top to bottom (*Run → Run All Cells*). Each notebook is independent and can be run in any order. To run a notebook from the command line instead:

```bash
jupyter nbconvert --to notebook --execute --inplace Neural_Network.ipynb
```

Approximate run times on a laptop CPU: `Neural_Network.ipynb` about 5 minutes, `Clustering.ipynb` about 6 minutes. `Random_Forest.ipynb` is the slowest notebook, because the 100-tree forests take several minutes each.

## Data preparation and validation

* **Train/validation split:** every classifier notebook splits each training file into 80% training and 20% validation with `train_test_split(..., test_size=0.2, stratify=y, random_state=42)`. All notebooks use the same split, so their validation accuracies can be compared directly. The test files are never used for model selection.
* **Standardization:** where it is used (k-NN experiment, Neural Network), the mean and standard deviation are computed on the training split only and then applied to the validation and test data. This avoids data leakage.
* **Clustering:** k-Means uses all 12,000 sentiment examples, standardized. The labels are used only after clustering, to evaluate the clusters.

## Training, evaluation and prediction

* **Training and evaluation:** each notebook trains its model on the training split and reports accuracy on the validation split. The hyperparameter experiments and the chosen values are explained in the notebook text.
* **Prediction:** the last section of each classifier notebook retrains the chosen model on all labelled examples, predicts the test files, and writes a submission file. The file is matched to `sample_submission.csv` by ID, not by row order.

## Validation results

| Method | Image validation accuracy | Sentiment validation accuracy |
|:--|--:|--:|
| k-NN | 53.08% (k = 1) | 57.46% (k = 20) |
| Decision Tree | 10.82% | 49.79% |
| Random Forest | 23.85% | 59.71% |
| Neural Network | 44.00% | 68.00% |

## Final Kaggle submission

We chose the best model for each task from the validation results:

* **Image:** k-NN with k = 1.
* **Sentiment:** Neural Network (two hidden layers of 256 and 64 units).

`Neural_Network.ipynb` (section 8) builds this combined file: `submission_knn_image_nn_sentiment.csv`. Public leaderboard score: 0.601.

## Reproducibility

* **Random seed:** 42. Every notebook uses it for the train/validation split. The Random Forest, Neural Network and k-Means notebooks also use it for their random steps (bootstrap samples, weight initialization, data shuffling, starting centroids).
* **Versions used for the Neural Network and Clustering notebooks:** Python 3.12.3, NumPy 2.5.5, pandas 3.1.1, scikit-learn 1.9.1, matplotlib 3.11.2, PyTorch 2.15.0 (CPU). The Neural Network, Clustering and Random Forest notebooks print the versions they were run with in their first or last cell.

## Acknowledgements

The use of AI tools and other external resources is listed in our report.
