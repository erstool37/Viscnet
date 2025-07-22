export PYTHONPATH=$PYTHONPATH:$(pwd)/src

python3 src/utils/preprocess_real.py -c configs/configclass.yaml -m real
python3 src/inference/viscometer.py -c configs/configclass.yaml -m real