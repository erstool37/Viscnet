python3 src/utils/preprocess.py -c configs/config1.yaml -m real
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python3 src/inference/viscometer.py -c configs/config1.yaml