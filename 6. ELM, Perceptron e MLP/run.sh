#!/bin/bash

echo 'ELM - Leave-One-Out - 10 Neurônios'
python3 rp_elm_loo_derm.py -q 10
echo '-------------------------------------------------------'
echo 'ELM - Leave-One-Out - 30 Neurônios'
python3 rp_elm_loo_derm.py -q 30
echo '-------------------------------------------------------'
echo 'ELM - Leave-One-Out - 50 Neurônios'
python3 rp_elm_loo_derm.py -q 50
echo '-------------------------------------------------------'
echo 'ELM - Leave-One-Out - 10 Neurônios - Dados normalizados'
python3 rp_elm_loo_derm.py -q 10 --normalize
echo '-------------------------------------------------------'
echo 'ELM - Leave-One-Out - 30 Neurônios - Dados normalizados'
python3 rp_elm_loo_derm.py -q 30 --normalize
echo '-------------------------------------------------------'
echo 'ELM - Leave-One-Out - 50 Neurônios - Dados normalizados'
python3 rp_elm_loo_derm.py -q 50 --normalize
echo '-------------------------------------------------------'
echo '-------------------------------------------------------'

echo 'Perceptron - Leave-One-Out - 1 Época'
python3 rp_perceptron_loo_derm.py -e 1
echo '-------------------------------------------------------'
echo 'Perceptron - Leave-One-Out - 10 Épocas'
python3 rp_perceptron_loo_derm.py -e 10
echo '-------------------------------------------------------'
echo 'Perceptron - Leave-One-Out - 20 Épocas'
python3 rp_perceptron_loo_derm.py -e 20
echo '-------------------------------------------------------'
echo 'Perceptron - Leave-One-Out - 1 Época - Dados normalizados'
python3 rp_perceptron_loo_derm.py -e 1 --normalize
echo '-------------------------------------------------------'
echo 'Perceptron - Leave-One-Out - 10 Épocas - Dados normalizados'
python3 rp_perceptron_loo_derm.py -e 10 --normalize
echo '-------------------------------------------------------'
echo 'Perceptron - Leave-One-Out - 20 Épocas - Dados normalizados'
python3 rp_perceptron_loo_derm.py -e 20 --normalize
echo '-------------------------------------------------------'
echo '-------------------------------------------------------'

echo 'MLP - Leave-One-Out - 10 Neurônios - 20 Épocas'
python3 rp_mlp_loo_derm.py -q 10
echo '-------------------------------------------------------'
echo 'MLP - Leave-One-Out - 30 Neurônios - 20 Épocas'
python3 rp_mlp_loo_derm.py -q 30
echo '-------------------------------------------------------'
echo 'MLP - Leave-One-Out - 50 Neurônios - 20 Épocas'
python3 rp_mlp_loo_derm.py -q 50
echo '-------------------------------------------------------'
echo 'MLP - Leave-One-Out - 10 Neurônios - 20 Épocas - Dados normalizados'
python3 rp_mlp_loo_derm.py -q 10 --normalize
echo '-------------------------------------------------------'
echo 'MLP - Leave-One-Out - 30 Neurônios - 20 Épocas - Dados normalizados'
python3 rp_mlp_loo_derm.py -q 30 --normalize
echo '-------------------------------------------------------'
echo 'MLP - Leave-One-Out - 50 Neurônios - 20 Épocas - Dados normalizados'
python3 rp_mlp_loo_derm.py -q 50 --normalize
echo '-------------------------------------------------------'
echo '-------------------------------------------------------'

echo 'ELM - 5-Fold - 10 Neurônios'
python3 rp_elm_kfold_derm.py -q 10
echo '-------------------------------------------------------'
echo 'ELM - 5-Fold - 30 Neurônios'
python3 rp_elm_kfold_derm.py -q 30
echo '-------------------------------------------------------'
echo 'ELM - 5-Fold - 50 Neurônios'
python3 rp_elm_kfold_derm.py -q 50
echo '-------------------------------------------------------'
echo 'ELM - 5-Fold - 10 Neurônios - Dados normalizados'
python3 rp_elm_kfold_derm.py -q 10 --normalize
echo '-------------------------------------------------------'
echo 'ELM - 5-Fold - 30 Neurônios - Dados normalizados'
python3 rp_elm_kfold_derm.py -q 30 --normalize
echo '-------------------------------------------------------'
echo 'ELM - 5-Fold - 50 Neurônios - Dados normalizados'
python3 rp_elm_kfold_derm.py -q 50 --normalize
echo '-------------------------------------------------------'
echo '-------------------------------------------------------'

echo 'Perceptron - 5-Fold - 1 Época'
python3 rp_perceptron_kfold_derm.py -e 1
echo '-------------------------------------------------------'
echo 'Perceptron - 5-Fold - 10 Épocas'
python3 rp_perceptron_kfold_derm.py -e 10
echo '-------------------------------------------------------'
echo 'Perceptron - 5-Fold - 20 Épocas'
python3 rp_perceptron_kfold_derm.py -e 20
echo '-------------------------------------------------------'
echo 'Perceptron - 5-Fold - 1 Época - Dados normalizados'
python3 rp_perceptron_kfold_derm.py -e 1 --normalize
echo '-------------------------------------------------------'
echo 'Perceptron - 5-Fold - 10 Épocas - Dados normalizados'
python3 rp_perceptron_kfold_derm.py -e 10 --normalize
echo '-------------------------------------------------------'
echo 'Perceptron - 5-Fold - 20 Épocas - Dados normalizados'
python3 rp_perceptron_kfold_derm.py -e 20 --normalize
echo '-------------------------------------------------------'
echo '-------------------------------------------------------'

echo 'MLP - 5-Fold - 10 Neurônios - 20 Épocas'
python3 rp_mlp_kfold_derm.py -q 10
echo '-------------------------------------------------------'
echo 'MLP - 5-Fold - 30 Neurônios - 20 Épocas'
python3 rp_mlp_kfold_derm.py -q 30
echo '-------------------------------------------------------'
echo 'MLP - 5-Fold - 50 Neurônios - 20 Épocas'
python3 rp_mlp_kfold_derm.py -q 50
echo '-------------------------------------------------------'
echo 'MLP - 5-Fold - 10 Neurônios - 20 Épocas - Dados normalizados'
python3 rp_mlp_kfold_derm.py -q 10 --normalize
echo '-------------------------------------------------------'
echo 'MLP - 5-Fold - 30 Neurônios - 20 Épocas - Dados normalizados'
python3 rp_mlp_kfold_derm.py -q 30 --normalize
echo '-------------------------------------------------------'
echo 'MLP - 5-Fold - 50 Neurônios - 20 Épocas - Dados normalizados'
python3 rp_mlp_kfold_derm.py -q 50 --normalize