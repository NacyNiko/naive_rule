# Naive Rules

This repository contains the implementation code for Naive Rules, a method for analyzing data based on the modified ICEWS14 and ICEWS05-15 datasets that have been resegmented to eliminate leakage issues.

## Installation

To use this code, please follow the steps below:

1. Clone the repository to your local machine:<br>
   `git clone https://github.com/NacyNiko/naive_rule.git`
2. Install the required dependencies. We recommend using a virtual environment to keep the dependencies isolated. Navigate to the project directory and run the following command:
   pip install -r requirements.txt
   
## Usage
Follow the steps below to run the Naive Rules code:

1. Run the `process_icews.py` script to preprocess the dataset. This script will perform necessary data transformations.
   python process_icews.py
2. Next, run the `model_rule_rhs.py` and `model_rule_lhs.py` scripts separately. These scripts calculate the scores for rhs and lhs, respectively. Make sure to specify the dataset as either `ICEWS14RR` or `ICEWS15RR` using the `--dataset` flag.
   python model_rule_rhs.py --dataset ICEWS14RR
   python model_rule_lhs.py --dataset ICEWS14RR
   
