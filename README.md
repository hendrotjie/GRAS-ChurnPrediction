# GRAS: A Hybrid Feature Selection Approach for Customer Churn Prediction

This repository contains the source code and experiment setup for the paper:

**"An Enhanced Feature Selection Technique for Churn Prediction using Hybrid Gravitational Search Algorithm and Simulated Annealing"**

## 🧠 Overview

The proposed GRAS method combines the global search capability of the **Gravitational Search Algorithm (GSA)** with the local refinement ability of **Simulated Annealing (SA)** to perform effective feature selection in customer churn prediction.

We evaluate the GRAS approach across four benchmark churn datasets and compare its performance against traditional, hybrid, and ensemble-based models.

## 📂 Repository Structure
├── src/ # Source code for GRAS implementation │ ├── gras.py # Main GRAS algorithm │ ├── gsa.py # Gravitational Search Algorithm │ ├── sa.py # Simulated Annealing │ └── utils.py # Helper functions ├── datasets/ # Public benchmark datasets (except Dataset 3, hosted externally) ├── results/ # Evaluation metrics and logs ├── notebooks/ # Jupyter notebooks for testing and visualization ├── requirements.txt # Python dependencies └── README.md # This file


## 🧪 Reproducing the Results
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/GRAS-ChurnPrediction.git
   cd GRAS-ChurnPrediction
2. Install Dependencies:
   pip install -r requirements.txt
3. Run the experiment:
   run the 02_Simulated_Annealing_Actual_GSA_EDITED.ipnyb file

📊 Datasets

This study uses four benchmark datasets. Due to GitHub file size restrictions, Dataset 3 (Cell2Cell) is hosted externally.

    Dataset 1: Telco Churn Data (3333 samples)
    Dataset 2: Telco Extended Dataset (7043 samples)
    Dataset 3: Cell2Cell Dataset (51047 samples)
    📥 Download link: https://drive.google.com/drive/folders/1bxRZLgY83PvXo2_l0PsQGdWEYBgZfjCD?usp=sharing
    Dataset 4: Telecom Churn Dataset (11 features)

Please place all datasets in the /datasets folder before running the code.  

📈 Evaluation Metrics

    Accuracy
    Precision
    Recall
    F1 Score
    AUC
    Overall Features Selected (OFS)

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

📬 Contact
For questions or collaboration:
    📧 Your Name: hendro@widyadharma.ac.id
    📄 Paper DOI: (Add once available)
