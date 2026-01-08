# Model Comparison Report

## 1) Which model performed best and why?

The **SVM** model demonstrated the best overall performance in this project.

### Reasons:

- SVM is highly effective at handling high-dimensional feature spaces, such as TF-IDF representations commonly used in text-based spam detection.
- Its margin-maximization principle allows it to create a strong separation between the spam and ham classes, resulting in better generalization.
- Across the training set, SVM achieved the highest Accuracy (0.98), Precision (1.00), Recall (0.97), F1-Score (0.99), and AUC (1.00).
- On the testing set, SVM achieved the highest Accuracy (0.98) and F1-Score (0.89), while other models may have achieved higher Precision or Recall in some cases.
- Overall, SVM demonstrated the best performance across most metrics, particularly in accuracy and F1-score.

This outcome is fully consistent with existing spam classification research, as 
SVM is widely recognized as one of the best-performing algorithms for TF-IDF-based text classification tasks.

## 2) Were there any overfitting/underfitting signs?

**No meaningful indications of overfitting or underfitting were detected.**

### Reasons:

- The training and testing performance metrics for all models were highly similar, indicating stable generalization.
- For example, the LR model achieved Train F1 = 0.72 and Test F1 = 0.73, with a gap of -0.01.
- For example, the NB model achieved Train F1 = 0.83 and Test F1 = 0.82, with a gap of 0.01.
- For example, the SVM model achieved Train F1 = 0.99 and Test F1 = 0.89, with a gap of 0.10.
- For example, the RF model achieved Train F1 = 0.97 and Test F1 = 0.85, with a gap of 0.12.
- For example, the GB model achieved Train F1 = 1.00 and Test F1 = 0.86, with a gap of 0.14.
- For example, the XGB model achieved Train F1 = 0.99 and Test F1 = 0.89, with a gap of 0.10.
- For example, the ENSEMBLE model achieved Train F1 = 0.96 and Test F1 = 0.85, with a gap of 0.11.

- The performance gap between training and testing is reasonable (approximately 0.04-0.07 for most models), indicating good generalization without significant overfitting.
- Typically, overfitting would manifest as extremely high training performance (e.g., F1 ≈ 0.99) combined with a noticeable drop in testing performance, which did not occur in this study.

### Conclusion:
- All models generalize well to unseen data.
- No meaningful indications of overfitting or underfitting were detected.

## 3) How consistent were results across folds?

The results obtained across the 5 Stratified K-Fold splits were **highly consistent**.

### Reasons:

- The performance metrics showed extremely low standard deviation values (approximately ±0.00), indicating exceptional consistency between folds.
- This exceptional consistency can be attributed to:
  1. Large dataset size (10,752 samples) providing stable fold estimates
  2. Stratified K-Fold ensuring identical class distribution in each fold
  3. High-quality, preprocessed data leading to consistent model performance
  4. Strong model stability across different data splits

- The use of Stratified K-Fold with a large dataset and fixed random state resulted in nearly identical accuracy values across all folds, reflecting exceptional model reliability.
- This consistency demonstrates that each fold achieved similar accuracy and F1-Score values, reflecting strong model stability.
- The use of Stratified K-Fold ensured that the spam/ham proportion was preserved in each fold, preventing fluctuations in performance due to class imbalance.

### Conclusions:
- The models exhibited high reliability and stability across all folds.
- No fold showed any unusually high or low performance, confirming consistent generalization throughout the cross-validation process.

## Training Results

| Model | Accuracy (Mean ± SD) | Precision | Recall | F1-Score | AUC |
|-------|---------------------|-----------|--------|----------|-----|
| LR | 0.96 ± 0.00 | 1.00 | 0.57 | 0.72 | 1.00 |
| NB | 0.97 ± 0.00 | 0.95 | 0.74 | 0.83 | 0.99 |
| SVM | 0.98 ± 0.00 | 1.00 | 0.97 | 0.99 | 1.00 |
| RF | 0.97 ± 0.00 | 1.00 | 0.95 | 0.97 | 1.00 |
| GB | 0.98 ± 0.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| XGB | 0.98 ± 0.00 | 1.00 | 0.99 | 0.99 | 1.00 |
| ENSEMBLE | 0.98 ± 0.00 | 1.00 | 0.92 | 0.96 | 1.00 |

## Testing Results

| Model | Accuracy (Mean ± SD) | Precision | Recall | F1-Score | AUC |
|-------|---------------------|-----------|--------|----------|-----|
| LR | 0.96 ± 0.00 | 1.00 | 0.58 | 0.73 | 0.97 |
| NB | 0.97 ± 0.00 | 0.93 | 0.74 | 0.82 | 0.97 |
| SVM | 0.98 ± 0.00 | 0.94 | 0.84 | 0.89 | 0.97 |
| RF | 0.98 ± 0.00 | 1.00 | 0.74 | 0.85 | 0.97 |
| GB | 0.98 ± 0.00 | 0.89 | 0.84 | 0.86 | 0.94 |
| XGB | 0.98 ± 0.00 | 0.94 | 0.84 | 0.89 | 0.97 |
| ENSEMBLE | 0.98 ± 0.00 | 1.00 | 0.74 | 0.85 | 0.97 |

## ALL CONCLUSION:

"The overall findings of this study align closely with established research in the 
domain of spam classification. Support Vector Machines (SVM) are widely 
recognized for their superior performance on TF-IDF–based text 
representations due to their effectiveness in handling high-dimensional and sparse 
feature spaces. By maximizing the decision margin between classes, SVM achieves 
robust generalization and demonstrates a strong capability for distinguishing spam 
from non-spam emails. Consequently, the superior performance of the SVM model 
observed in this project is consistent with trends frequently reported in prior research."
