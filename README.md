# Naive Rules

This repository contains the implementation code for Naive Rules, a method for analyzing data based on the modified ICEWS14 and ICEWS05-15 datasets that have been resegmented to eliminate leakage issues. The paper link is [Do Temporal Knowledge Graph Embedding Models Learn or Memorize Shortcuts?](https://openreview.net/forum?id=UMokRwWfLW&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2023%2FWorkshop%2FTGL%2FAuthors%23your-submissions))
## Installation

To use this code, please follow the steps below:

1. Clone the repository to your local machine:<br>
   `git clone https://github.com/NacyNiko/naive_rule.git`

   
## Usage
Follow the steps below to run the Naive Rules code:

1. Run the `process_icews.py` script to preprocess the dataset. This script will perform necessary data transformations.<br>
   `python process_icews.py`
2. Next, run the `model_rule_rhs.py` and `model_rule_lhs.py` scripts separately. These scripts calculate the scores for rhs and lhs, respectively. Make sure to specify the dataset as either `ICEWS14RR` or `ICEWS15RR` using the `--dataset` flag.<br>
   `python model_rule_rhs.py --dataset ICEWS14RR`<br>
   `python model_rule_lhs.py --dataset ICEWS14RR`
3. The averange of rhs score and lhs score was used as the final score.
   
