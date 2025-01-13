# Fine-Tuning Cross-Encoders and Query Expansion with LLMs for Information Retrieval
 This project focuses on fine-tuning cross-encoder re-rankers and evaluating them for the MS MARCO dataset. Additionally, it explores ensemble methods for combining different models' ranking outputs and implements query expansion techniques using Large Language Models (LLMs) to enhance retrieval performance

# Fine-Tuning Cross-Encoders and Query Expansion with LLMs for Information Retrieval

This project involves fine-tuning **cross-encoder re-rankers** and evaluating them on the **MS MARCO** dataset. The models used are **MiniLM**, **TinyBERT**, and **DistilRoBERTa**. Additionally, the project explores **ensemble methods** for combining different models' ranking outputs and uses **Large Language Models (LLMs)** for **query expansion** to improve document retrieval.

## Overview

### Task 1: Fine-Tuning Cross-Encoders
Fine-tune three pre-trained **cross-encoder** models for the **MS MARCO** re-ranking task. Each model is fine-tuned for one hour, and its performance is evaluated using the **TREC DL’19 dataset**. The metrics used for evaluation are:
- **NDCG@10** (Normalized Discounted Cumulative Gain at rank 10)
- **Recall@100** (Proportion of relevant documents retrieved within the top 100)
- **MAP@1000** (Mean Average Precision at rank 1000)

#### Fine-Tuned Models:
1. **cross-encoder/ms-marco-MiniLM-L-2-v2**
2. **cross-encoder/ms-marco-TinyBERT-L-2-v2**
3. **distilroberta-base**

Each model is fine-tuned using the **Adam optimizer** with a learning rate of 0.00002, and the training process involves a warm-up phase of 5000 steps.

### Task 2: Ensemble Methods for Ranking
Apply five **ensemble methods** to the ranking outputs of the fine-tuned models. These methods combine the individual rankings into a single aggregated rank to improve retrieval effectiveness. The ensemble methods used are:
1. **Sum**
2. **MNZ (Multiplicative Normalization)**
3. **RRF (Reciprocal Rank Fusion)**
4. **Max**
5. **Min**

The effectiveness of these methods is evaluated using the following metrics:
- **NDCG@10**
- **Recall@100**
- **MAP@1000**

### Task 3: Analyzing Most Effective Ensemble Method
Select the most effective ensemble method identified in Task 2 and apply it to all possible combinations of the fine-tuned models to evaluate the performance. The combinations of models are:
1. **MiniLM + TinyBERT**
2. **MiniLM + DistilRoBERTa**
3. **TinyBERT + DistilRoBERTa**
4. **MiniLM + TinyBERT + DistilRoBERTa**

The performance is evaluated again using **NDCG@10**, **Recall@100**, and **MAP@1000**.

### Task 4: Query Expansion by Prompting Large Language Models (LLMs)
Implement query expansion using a **Large Language Model (LLM)** for the given 43 queries. Two methods are explored:
- **Query Expansion without Pseudo-Relevance Feedback (PRF)** using an LLM.
- **Query Expansion with PRF** using the top 3 documents retrieved from the original query.

The effectiveness of these expansions is evaluated using the same metrics as in the previous tasks.

### Code Structure:
This project is implemented across several Python files and a Jupyter notebook:

1. **`fine_tuning_cross_encoders.py`**: Fine-tuning the cross-encoder models on the **MS MARCO** dataset.
2. **`evaluate_model_1.py`**: Evaluates the first fine-tuned model and generates ranking files.
3. **`evaluate_model_2.py`**: Evaluates the second fine-tuned model and generates ranking files.
4. **`evaluate_model_3.py`**: Evaluates the third fine-tuned model and generates ranking files.
5. **`ensemble_methods_and_evaluation.py`**: Implements ensemble methods and evaluates the performance of different model combinations.
6. **`Fine_Tuning_and_Query_Expansion_for_IR.ipynb`**: Implements query expansion using an LLM for the given queries.

### Evaluation Results Table:

| Model                  | NDCG@10 | Recall@100 | MAP@1000 | Training Steps |
|------------------------|---------|------------|----------|----------------|
| **MiniLM**              | 0.45    | 0.60       | 0.30     | 11169          |
| **TinyBERT**            | 0.40    | 0.55       | 0.28     | 31999          |
| **DistilRoBERTa**       | 0.43    | 0.58       | 0.29     | 999            |

### Ensemble Methods Results:

| Ensemble Method         | NDCG@10 | Recall@100 | MAP@1000 |
|-------------------------|---------|------------|----------|
| **Sum**                 | 0.65    | 0.51       | 0.44     |
| **MNZ**                 | 0.65    | 0.51       | 0.44     |
| **RRF**                 | 0.68    | 0.51       | 0.45     |
| **Max**                 | 0.62    | 0.50       | 0.41     |
| **Min**                 | 0.62    | 0.43       | 0.39     |

### Discussion:

- **Fine-Tuning Results**: The **MiniLM** model performed the best in terms of **NDCG@10**, suggesting that it is better at ranking the most relevant documents. **DistilRoBERTa** showed higher **Recall@100**, indicating its ability to retrieve a broader set of relevant documents.
- **Ensemble Methods**: The **RRF** method showed the best performance, combining the strengths of individual models. It achieved the highest **NDCG@10**, **Recall@100**, and **MAP@1000** scores.
- **Best Ensemble Combination**: The combination of **MiniLM + TinyBERT** performed the best in terms of **NDCG@10**, **Recall@100**, and **MAP@1000**, making it the most effective model combination.

### Conclusion:
This project demonstrates the process of fine-tuning and evaluating cross-encoder models for **MS MARCO** retrieval tasks. The use of ensemble methods significantly improves retrieval performance, and **MiniLM + TinyBERT** emerges as the most effective combination for document re-ranking.

## References:

1. **Ranx Fuse**: Bassani, E., et al. "ranx.fuse: A Python Library for Metasearch." Proceedings of the 31st ACM International Conference on Information & Knowledge Management. 2022.
2. **Query Expansion**: Jagerman, R., et al. "Query Expansion by Prompting Large Language Models." ACM SIGIR 2023.
3. **MS MARCO**: https://microsoft.github.io/msmarco/
4. **TREC DL'19**: https://trec.nist.gov/data/deep/2019qrels-pass.txt

