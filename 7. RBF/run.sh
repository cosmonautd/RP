#!/bin/bash
echo 'RBF Leave One Out sem normalização'
python3 rp_rbf_loo.py
echo '--'
echo 'RBF Leave One Out com normalização'
python3 rp_rbf_loo.py --normalize
echo '--'
echo 'RBF 10-Fold sem normalização'
python3 rp_rbf_kfold.py
echo '--'
echo 'RBF 10-Fold com normalização'
python3 rp_rbf_kfold.py --normalize
echo '--'
echo 'RBF Holdout (70/30/20rep) sem normalização'
python3 rp_rbf_holdout.py
echo '--'
echo 'RBF Holdout (70/30/20rep) com normalização'
python3 rp_rbf_holdout.py --normalize