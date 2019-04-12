#!/bin/bash
echo 'ELM Leave One Out sem normalização'
python3 rp_elm_loo.py
echo '--'
echo 'ELM Leave One Out com normalização'
python3 rp_elm_loo.py --normalize
echo '--'
echo 'ELM 10-Fold sem normalização'
python3 rp_elm_kfold.py
echo '--'
echo 'ELM 10-Fold com normalização'
python3 rp_elm_kfold.py --normalize
echo '--'
echo 'ELM Holdout (70/30/20rep) sem normalização'
python3 rp_elm_holdout.py
echo '--'
echo 'ELM Holdout (70/30/20rep) com normalização'
python3 rp_elm_holdout.py --normalize