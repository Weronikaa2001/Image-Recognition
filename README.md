# Image-Recognition-Garbage-Classification

## Project Overview

This project focuses on developing a deep learning system for automatic garbage classification. The goal is to classify waste images into different categories and investigate how image resolution affects model performance and generalization ability.

The project applies transfer learning using **EfficientNet-B0** and evaluates performance through cross-validation, real-world testing, and unseen hold-out data.

---

## Business Context

Proper waste segregation is important for improving recycling efficiency and reducing environmental impact. Incorrect waste disposal can increase processing costs and reduce the effectiveness of recycling systems. Manual sorting is often time-consuming and prone to human error.

An automated image classification system could support smart waste management solutions such as mobile applications, smart bins, and automated recycling systems by assisting users in identifying the correct waste category.

---

## Categories

The model classifies garbage into the following 11 categories:

- Aluminium  
- Batteries  
- Cardboard  
- Disposable Plates  
- Glass  
- Hard Plastic  
- Paper  
- Paper Towel  
- Polystyrene  
- Soft Plastics  
- Takeaway Cups  

---

## Data Augmentation

To improve model robustness and reduce overfitting, several augmentation techniques were applied during training:

- Random horizontal flip
- Random rotation
- Brightness adjustment
- Contrast adjustment
- Saturation changes
- Hue modifications

These transformations allowed the model to learn from slightly different versions of the same objects.

---

## Model Selection

The project uses:

```python
model_name = "google/efficientnet-b0"
```

EfficientNet-B0 was selected because it provides a good balance between accuracy and computational efficiency.

---

## Cloud Training Setup

Training with **5-fold cross-validation** and multiple image resolutions required substantial computational resources. Due to long execution times and hardware limitations, cloud resources were used.

To simplify cloud execution, the notebook workflow was converted into:

```text
Trash_classification.py
```

This allowed long training jobs to run more efficiently and improved reproducibility.

---

## Training Strategy

The model was trained using:

- **5-fold Stratified Cross-Validation**
- **15 epochs**
- Batch size: `16`
- Learning rate: `5e-5`

For each fold:

- Four folds were used for training
- One fold was used for validation
- The process repeated five times

The best-performing model across folds was saved as:

```text
./best_garbage_classifier
```

---

## Saved Outputs

Training outputs and checkpoints were stored separately for each fold.

### 224×224 experiment

```text
garbage_model_fold_1
garbage_model_fold_2
garbage_model_fold_3
garbage_model_fold_4
garbage_model_fold_5
```

### 448×448 experiment

```text
garbage_model_448_fold_1
garbage_model_448_fold_2
garbage_model_448_fold_3
garbage_model_448_fold_4
garbage_model_448_fold_5
```

These folders contain:

- model checkpoints
- training logs
- evaluation metrics
- configuration files

---

## Experiments

Two image resolutions were compared:

### 224×224

- More stable validation performance
- Better generalization
- Selected as the final model

### 448×448

- Lower training loss
- Signs of overfitting
- Less stable across folds

---

## Testing Strategy

Testing was performed in two phases.

### Phase 1 — Real-world images

Images were collected during everyday waste disposal situations and used to evaluate performance in realistic conditions.

These images introduced:

- different backgrounds
- lighting changes
- varying object positions
- real-world variability

### Phase 2 — Hold-out Test Vault

The final model was evaluated using the previously separated **15% unseen test set**.

No retraining was performed.

The saved model was loaded and used only for inference.

---

## Results

Final evaluation on unseen data produced:

**Final Test Accuracy: 70.8%**

Key observations:

- The model successfully learned meaningful visual patterns
- Similar categories remained difficult to distinguish
- Real-world images produced lower confidence scores
- The 224×224 model generalized better than 448×448

---

## Future Improvements

Potential future improvements include:

- Increasing dataset size
- Improving class balance
- Adding more real-world examples
- Testing additional architectures
- Expanding difficult categories

---

## Authors

**Weronika Mądro**
**Yuliya Martyniuk**
