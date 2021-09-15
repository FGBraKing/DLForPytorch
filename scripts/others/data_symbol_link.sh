#!/bin/bash
set -ex

# All server for project.
SRC_DATASET="/data/project_data_lf/DLForPytorch/datasets"
AIM_DATASET="/home/users/lf/CODE/PycharmProjects/DLForPytorch/traces/"

SRC_CHECKPOINTS="/data/project_data_lf/DLForPytorch/checkpoints"
AIM_CHECKPOINTS="/home/users/lf/CODE/PycharmProjects/DLForPytorch/traces/"

SRC_LOGS="/data/project_data_lf/DLForPytorch/logs"
AIM_LOGS="/home/users/lf/CODE/PycharmProjects/DLForPytorch/traces/"

SRC_RESULTS="/data/project_data_lf/DLForPytorch/results"
AIM_RESULTS="/home/users/lf/CODE/PycharmProjects/DLForPytorch/traces/"

ln -s ${SRC_DATASET} ${AIM_DATASET}
ln -s ${SRC_CHECKPOINTS} ${AIM_CHECKPOINTS}
ln -s ${SRC_LOGS} ${AIM_LOGS}
ln -s ${SRC_RESULTS} ${AIM_RESULTS}



test_src="/data/project_data_lf/DLForPytorch/datasets
/data/project_data_lf/DLForPytorch/checkpoints
/data/project_data_lf/DLForPytorch/logs
/data/project_data_lf/DLForPytorch/results"

test_aim="/home/users/lf/CODE/PycharmProjects/DLForPytorch/traces/datasets
/home/users/lf/CODE/PycharmProjects/DLForPytorch/traces/logs
/home/users/lf/CODE/PycharmProjects/DLForPytorch/traces/checkpoints
/home/users/lf/CODE/PycharmProjects/DLForPytorch/traces/results"

src_num=$(echo ${test_src} | wc -w)
aim_num=$(echo ${test_aim} | wc -w)
echo ${src_num}
echo ${aim_num}

if [ ${src_num} -ne ${aim_num} ]; then
  echo "${src_num} -ne ${aim_num}"
  exit 1
fi

for src in ${test_src}
do
  echo ${src}
done



