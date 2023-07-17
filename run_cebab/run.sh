# train cbm
python cbm_joint_sparse_obert.py --model_name distilbert-base-uncased --FINAL_SPARSITY 0.7 > log/joint/0.2_0.7_distillbert.txt

# test-time intervention
python cbm_joint_obert_test_adaptation.py --MASK_PORTION 0.3