export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python3 src/utils/preprocess.py -c configs/config0.yaml -m preprocess
python3 src/inference/viscometer.py -c configs/config0.yaml # error calculation
python3 src/inference/PCA.py -c configs/config0.yaml # PCA for rpm, viscosity