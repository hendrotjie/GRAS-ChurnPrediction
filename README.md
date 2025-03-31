# GRAS: A Hybrid Feature Selection Approach for Customer Churn Prediction

This repository contains the source code and experiment setup for the paper:

**"An Enhanced Feature Selection Technique for Churn Prediction using Hybrid Gravitational Search Algorithm and Simulated Annealing"**

## 🧠 Overview

The proposed **GRAS** method combines the global search capability of the **Gravitational Search Algorithm (GSA)** with the local refinement strength of **Simulated Annealing (SA)** to perform effective feature selection for customer churn prediction.

We evaluate GRAS across four benchmark churn datasets and compare its performance against traditional, hybrid, and ensemble-based methods.

---

## 📂 Repository Structure

```
├── src/                # Source code for GRAS
│   ├── _init_.py         
│   ├── GSA_implementation.py          
│   ├── benchmarks.py                 
│   ├── generate_neighbor.py           
│   ├── gsa_sa_iterative.py            
│   ├── main.py                        
│   ├── solution.py                    
│   └── utils.py        
├── datasets/           # Benchmark datasets (except Dataset 3)
├── results/            # Output metrics and logs
├── notebooks/          # Jupyter notebooks for testing
│   └── 02_Simulated_Annealing_Acutal_GSA_EDITED.ipynb
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## 🧪 Reproducing the Results

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hendrotjie/GRAS-ChurnPrediction.git
   cd GRAS-ChurnPrediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the experiment:**
   Open and execute the notebook:

   ```
   notebooks/02_Simulated_Annealing_Actual_GSA_EDITED.ipynb
   ```

---

## 📊 Datasets

This study uses four benchmark datasets:

| Dataset | Description                      | Size     |
|---------|----------------------------------|----------|
| Dataset 1 | Telco Churn Data                 | 3,333 samples |
| Dataset 2 | Telco Extended Dataset           | 7,043 samples |
| Dataset 3 | Cell2Cell (hosted externally)    | 51,047 samples |
| Dataset 4 | Telecom Churn Dataset (11 features) | Small-scale |

⚠️ **Due to GitHub file size restrictions**, Dataset 3 is hosted externally.  
📥 Download it from:  
[https://drive.google.com/drive/folders/1bxRZLgY83PvXo2_l0PsQGdWEYBgZfjCD](https://drive.google.com/drive/folders/1bxRZLgY83PvXo2_l0PsQGdWEYBgZfjCD?usp=sharing)

**Place all datasets inside the `datasets/` folder** before running the experiments.

---

## 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- AUC
- Overall Features Selected (OFS)

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 📬 Contact

- **Name:** Hendro
- **Email:** hendro@widyadharma.ac.id  
- **Paper DOI:** _Coming soon_
