#!/bin/bash

# main_name = "main_baseline.py"

# 遍历 word dataset
for patient_id in {1..10}; do
    patient="sub-$(printf '%02d' $patient_id)"
    echo "Training on word dataset, patient: $patient"
    python ./src/main_baseline.py --dataset='word' --patient=$patient
    echo "Training completed for word dataset, patient: $patient"
    echo "------------------------------"
done

# 遍历 sentence dataset
# for patient_id in {1..3}; do
#     patient="p$(printf '%d' $patient_id)"
#     echo "Training on sentence dataset, patient: $patient"
#     python ./src/main_baseline.py --dataset='sentence' --patient=$patient
#     echo "Training completed for sentence dataset, patient: $patient"
#     echo "------------------------------"
# done

echo "All training complete."/