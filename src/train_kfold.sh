#!/bin/bash

# 遍历 word dataset
for patient_id in {1..10}; do
    for fold in {0..9}; do
        patient="sub-$(printf '%02d' $patient_id)"
        echo "Training on word dataset, patient: $patient, fold: $fold"
        python ./src/main_multiscale_CBAM_hidden512_latent1024.py --dataset='word' --patient=$patient --fold=$fold
        echo "Training completed for word dataset, patient: $patient, fold: $fold"
        echo "------------------------------"
    done
done


# 遍历 sentence dataset
# for patient_id in {1..3}; do
#     patient="p$(printf '%d' $patient_id)"
#     for fold in {0..9}; do
#         echo "Training on sentence dataset, patient: $patient, fold: $fold"
#         python ./src/main_multiscale_sa_v3.py --dataset='sentence' --patient=$patient --fold=$fold
#         echo "Training completed for sentence dataset, patient: $patient, fold: $fold"
#         echo "------------------------------"
#     done
# done

echo "All training complete."/