export CUBLAS_WORKSPACE_CONFIG=:4096:8

python3 src/utils/preprocess.py -c configs/config.yaml -m train
# python3 src/utils/preprocess.py -c configs/config.yaml -m real
# python3 src/main.py -c configs/config.yaml
python3 src/main_rpm.py -c configs/config.yaml