#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !pip install datasets
# !pip install imagehash
# !pip install torchvision
# !pip install transformers
# !pip install evaluate
# !pip install transformers[torch]
#!pip install "accelerate>=1.1.0" 


# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datasets import load_dataset
from huggingface_hub import login
from collections import Counter
from PIL import Image
import imagehash
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomRotation, ColorJitter
import evaluate
import torch
from sklearn.model_selection import StratifiedKFold
import gc
import os
import json


# In[ ]:


# login(token="") 
# dataset = load_dataset("viola77data/recycling-dataset",num_proc=8)
# data = dataset["train"] # the author named the whole dataset as train, the 'train' split contains all the data in this case

# total_images = len(data)
# print(f"Loaded {total_images:,} images locally.\n")


# In[2]:


# data_dir = r"C:\Users\ydmar\.cache\huggingface\hub\datasets--viola77data--recycling-dataset\snapshots\e2e03c91c385e8d1a758389cdb20cf9c024f6cbf"
data_dir = "./trash_images"
dataset = load_dataset("imagefolder", data_dir=data_dir)
data = dataset["train"]
split_data = data.train_test_split(test_size=0.15, stratify_by_column="label", seed=42)
cv_train = split_data["train"]      # The 85% we will use for the 5-Fold Loop
test = split_data["test"]

print(f"Total images: {len(data)}")
print(f"Images for Cross-Validation: {len(cv_train)}")
print(f"Images locked in the Test Vault: {len(test)}")


# In[11]:


labels = cv_train.features["label"].names
id2label = {str(i): c for i, c in enumerate(labels)}
label2id = {c: str(i) for i, c in enumerate(labels)}
all_numerical_labels = cv_train["label"]

print(f"Classes found: {labels}")


# ## Setup and training for the default image resolution (224x224)

# In[12]:


model_name = "google/efficientnet-b0" 
image_processor = AutoImageProcessor.from_pretrained(model_name)


# In[13]:


augmentations = Compose([
    RandomHorizontalFlip(),
    RandomRotation(20),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
])

# Transform for TRAINING with augmentations
def train_transforms(batch):
    augmented_images = [augmentations(x.convert("RGB")) for x in batch["image"]] #  Apply the random flips and colors to the raw PIL images
    inputs = image_processor(augmented_images, return_tensors="pt") # Resize, Tensor, and Normalize
    inputs["label"] = batch["label"]
    return inputs

# Transform for VALIDATION without augmentations
def val_transforms(batch):
    inputs = image_processor([x.convert("RGB") for x in batch["image"]], return_tensors="pt")
    inputs["label"] = batch["label"]
    return inputs

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }

# Metric calculation
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, true_labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=true_labels)


# In[ ]:


n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

fold_accuracies = [] 
best_overall_accuracy = 0.0  
best_fold = 0
oof_preds = []
oof_labels = []

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(all_numerical_labels)), all_numerical_labels)):
    print(f"\n================================")
    print(f"       STARTING FOLD {fold + 1}/{n_splits}")
    print(f"================================")

    # Create  train/val subsets for this round
    fold_train_data = cv_train.select(train_idx).with_transform(train_transforms)
    fold_val_data = cv_train.select(val_idx).with_transform(val_transforms)


    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=len(labels), # num_labels is set to the number of classes in our dataset
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True  # allows loading pretrained weights even if the classifier head size doesn’t match your number of classes
    )

    # Setup Trainer for this fold
    training_args = TrainingArguments(
        output_dir=f"./garbage_model_fold_{fold + 1}", # Save each fold in a separate folder
        remove_unused_columns=False, # preventing deleting images
        eval_strategy="epoch", # Evaluate at the end of every epoch
        save_strategy="epoch", # Save model at the end of every epoch
        learning_rate=5e-5,
        per_device_train_batch_size=16, # batch size for training
        gradient_accumulation_steps=2, # accumulate gradients over 2 steps to effectively have a batch size of 64 without OOM errors
        per_device_eval_batch_size=16, # batch size for test
        num_train_epochs=15, # 3  number of training epochs
        warmup_ratio=0.1, # 10% of training steps will be used for a linear warmup of the learning rate
        logging_steps=10, # controls how often the training process prints out progress or metrics
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=fold_train_data,
        eval_dataset=fold_val_data,
        processing_class=image_processor,
        compute_metrics=compute_metrics,
    )
    trainer.train() # Train the Fold

    # Get final validation score per fold
    prediction_output = trainer.predict(fold_val_data)
    fold_acc = prediction_output.metrics['test_accuracy'] 
    fold_accuracies.append(fold_acc)
    print(f"\n>>> Fold {fold + 1} Accuracy: {fold_acc * 100:.2f}%\n")

    # Grab the winning guesses (argmax) and the true answers
    batch_preds = np.argmax(prediction_output.predictions, axis=1)
    batch_labels = prediction_output.label_ids

    # Store them in our global OOF lists
    oof_preds.extend(batch_preds)
    oof_labels.extend(batch_labels)

    if fold_acc > best_overall_accuracy:
        print(f"Best model so far found in Fold {fold + 1} (Accuracy: {fold_acc * 100:.2f}%). Saving...")
        best_overall_accuracy = fold_acc
        best_fold = fold + 1
        trainer.save_model("./best_garbage_classifier")
        image_processor.save_pretrained("./best_garbage_classifier")

    # Free up computer memory before starting the next fold
    del model, trainer, training_args
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


print("\n================================")
print("      CROSS-VALIDATION COMPLETE")
print("================================")
for i, acc in enumerate(fold_accuracies):
    print(f"Fold {i + 1}: {acc * 100:.2f}%")
print(f"\nAverage Accuracy across all {n_splits} folds: {np.mean(fold_accuracies) * 100:.2f}%")
print(f"Standard Deviation: ±{np.std(fold_accuracies) * 100:.2f}%")

print(f"\nThe best model was Fold {best_fold} with {best_overall_accuracy * 100:.2f}% accuracy.")
print("It has been saved to the folder: './best_garbage_classifier'")


# In[18]:


from transformers.utils.notebook import NotebookProgressCallback


# In[19]:


best_model = AutoModelForImageClassification.from_pretrained("./best_garbage_classifier")
best_processor = AutoImageProcessor.from_pretrained("./best_garbage_classifier")

test_prep = test.with_transform(val_transforms)

test_args = TrainingArguments(
    output_dir="./final_test_results",
    remove_unused_columns=False, 
    per_device_eval_batch_size=16,
    report_to="none" 
)

# Short evaluation
final_trainer = Trainer(
    model=best_model,
    args=test_args,
    data_collator=collate_fn,
    eval_dataset=test_prep,
    processing_class=best_processor,
    compute_metrics=compute_metrics,
)

final_trainer.remove_callback(NotebookProgressCallback)
print(f"Testing against {len(test_prep)} unseen vault images")
final_results = final_trainer.evaluate()
final_accuracy = final_results['eval_accuracy'] * 100
print(f"\n========================================")
print(f"Final test vault accuracy: {final_accuracy:.2f}%")


# ## Setup and training for the higher image resolution (448x448)



# In[23]:


image_processor = AutoImageProcessor.from_pretrained(model_name)
image_processor.size = {"height": 448, "width": 448}

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

fold_accuracies = [] 
best_overall_accuracy = 0.0  
best_fold = 0
oof_preds = []
oof_labels = []

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(all_numerical_labels)), all_numerical_labels)):
    print(f"\n================================")
    print(f"       STARTING FOLD {fold + 1}/{n_splits}")
    print(f"================================")

    # Create  train/val subsets for this round
    fold_train_data = cv_train.select(train_idx).with_transform(train_transforms)
    fold_val_data = cv_train.select(val_idx).with_transform(val_transforms)


    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=len(labels), # num_labels is set to the number of classes in our dataset
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True  # allows loading pretrained weights even if the classifier head size doesn’t match your number of classes
    )

    # Setup Trainer for this fold
    training_args = TrainingArguments(
        output_dir=f"./garbage_model_448_fold_{fold + 1}", 
        remove_unused_columns=False, 
        eval_strategy="epoch", 
        save_strategy="epoch", 
        learning_rate=5e-5,
        per_device_train_batch_size=8, # reduced from 16
        gradient_accumulation_steps=4,  # reduced from 4
        per_device_eval_batch_size=8,   # reduced from 16
        num_train_epochs=15,      
        warmup_ratio=0.1, 
        logging_steps=10, 
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=fold_train_data,
        eval_dataset=fold_val_data,
        processing_class=image_processor,
        compute_metrics=compute_metrics,
    )
    trainer.train() # Train the Fold

    # Get final validation score per fold
    prediction_output = trainer.predict(fold_val_data)
    fold_acc = prediction_output.metrics['test_accuracy'] 
    fold_accuracies.append(fold_acc)
    print(f"\n>>> Fold {fold + 1} Accuracy: {fold_acc * 100:.2f}%\n")

    # Grab the winning guesses (argmax) and the true answers
    batch_preds = np.argmax(prediction_output.predictions, axis=1)
    batch_labels = prediction_output.label_ids

    # Store them in our global OOF lists
    oof_preds.extend(batch_preds)
    oof_labels.extend(batch_labels)

    if fold_acc > best_overall_accuracy:
        print(f"Best 448px model so far found in Fold {fold + 1} (Accuracy: {fold_acc * 100:.2f}%). Saving...")
        best_overall_accuracy = fold_acc
        best_fold = fold + 1
        trainer.save_model("./model_448px_experiment")
        image_processor.save_pretrained("./model_448px_experiment")

    # Free up computer memory before starting the next fold
    del model, trainer, training_args
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


print("\n================================")
print("      CROSS-VALIDATION COMPLETE")
print("================================")
for i, acc in enumerate(fold_accuracies):
    print(f"Fold {i + 1}: {acc * 100:.2f}%")
print(f"\nAverage Accuracy across all {n_splits} folds: {np.mean(fold_accuracies) * 100:.2f}%")
print(f"Standard Deviation: ±{np.std(fold_accuracies) * 100:.2f}%")

print(f"\nThe best model was Fold {best_fold} with {best_overall_accuracy * 100:.2f}% accuracy.")
print("It has been saved to the folder: './model_448px_experiment'")



