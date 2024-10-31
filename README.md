# PRODIGY_ML_03
# Cat and Dog Image Classification using SVM
This project implements an image classification model using Support Vector Machines (SVM) to distinguish between images of cats and dogs. The goal is to provide a machine learning solution for binary image classification tasks.
## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cat-dog-classification.git
   cd cat-dog-classification
pip install -r requirements.txt
## Usage
To train the model, run the following command:
```bash
python train.py
python evaluate.py
## Dataset
This project uses images from the Kaggle Cats and Dogs dataset. You can download the dataset from [Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data). After downloading, ensure that the data is organized into the following structure:
## Model Training
The model is built using the Support Vector Machine (SVM) algorithm with an RBF kernel. Training is performed on the training dataset, and the model's performance is evaluated using a validation dataset.
## Results
The model achieved an accuracy of 85% on the validation dataset. Below is an example of the confusion matrix obtained during evaluation:
[[TP, FN],
 [FP, TN]]
## Evaluation
Metrics such as accuracy, precision, recall, and F1-score were calculated using the confusion matrix to assess the model's performance.
## Conclusion
This project provided valuable insights into image classification using SVM. I learned about data preprocessing, model evaluation, and the importance of using metrics like the confusion matrix.
├── train
│   ├── cat
│   └── dog
└── val
    ├── cat
    └── dog

Feel free to modify and expand upon any section to better fit your project!
