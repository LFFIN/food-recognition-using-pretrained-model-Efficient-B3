# food-recognition-using-pretrained-model-Efficient-B3


# 🍽 Food Recognition Challenge (Multi-Label Classification)

##  Overview

This repository contains my work for the **Food Recognition Challenge**, an in-class competition for a Machine Learning course.

* **Type:** Multi-label image classification (each picture can contain multiple foods)
* **Goal:** Train deep learning models that accurately predict food categories from real-world food images.
* **Timeline:**

  * **Start:** April 1, 2025
  * **End:** June 20, 2025


## 🗂 Dataset

The dataset is derived from the **AIcrowd Food Recognition Benchmark (2022)**.

* **Train set:** 39,962 images
* **Test set:** 1,000 images
* **Categories:** 498 food classes

Each image may contain **more than one category**.



##  Evaluation Metric

Submissions are evaluated using the **Micro F1-score**, comparing predicted labels with the ground truth.



## 📄 Submission Format

Your submission must be a CSV file with the following format:

* First column → **Filename** (test image name)
* Next 498 columns → **binary predictions (0/1)** for each food class

### Example


Filename,0,1,2,3,...,497
594245.jpg,0,0,1,0,...,0
605396.jpg,0,1,0,0,...,1
800684.jpg,0,0,0,0,...,0
470316.jpg,1,0,0,0,...,0
119343.jpg,0,0,0,0,...,1



## My Approach

I used **transfer learning** with a **pretrained EfficientNet-B3 model**, fine-tuned for this multi-label classification task.

### Key steps:

1. **Data Preprocessing**

   * Image resizing & normalization
   * Multi-label encoding from CSV

2. **Modeling**

   * Base model: **EfficientNet-B3** (ImageNet pretrained)
   * Final layer adapted to **498 output nodes** with `sigmoid` activation
   * Loss function: **Binary Cross-Entropy**
   * Optimizer: **AdamW**

3. **Training Strategy**

   * Multi-label stratified K-fold cross-validation
   * Augmentations (random flips, rotations, color jitter)
   * Threshold tuning for label selection

4. **Results**

   * Best performing model: **EfficientNet-B3**
   * Achieved strong performance with a competitive **Micro F1 score**


##  How to Run

1. Clone the repo:

   ```bash
   git clone https://github.com/yourusername/food-recognition-challenge.git
   cd food-recognition-challenge
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Train the model:

   ```bash
   python src/train.py
   ```

4. Generate predictions:

   ```bash
   python src/inference.py --weights models/efficientnet_b3.pth
   ```

5. Create submission file:

   ```bash
   python src/evaluate.py --predictions predictions.csv
   ```

---

##  Future Improvements

* Model ensembling (EfficientNet, ResNet, DenseNet)
* Advanced augmentation (Mixup, CutMix)
* Hyperparameter tuning
* Semi-supervised learning with pseudo-labels

---

##  Acknowledgements

* [AIcrowd Food Recognition Benchmark](https://www.aicrowd.com/challenges/food-recognition-benchmark)
* PyTorch team & open-source community

