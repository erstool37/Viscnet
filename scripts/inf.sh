export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# python3 src/utils/preprocess.py -c configs/configclass.yaml -m real
python3 src/inference/viscometer.py -c configs/configclass.yaml -m real # error calculation
# python3 src/inference/PCA.py -c configs/config.yaml -m PCA # PCA for rpm, viscosity