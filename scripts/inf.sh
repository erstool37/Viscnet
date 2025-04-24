python3 src/utils/preprocess.py -c configs/config0.yaml -m real
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python3 src/inference/viscometer.py -c configs/config0.yaml