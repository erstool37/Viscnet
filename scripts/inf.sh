export PYTHONPATH=$PYTHONPATH:$(pwd)/src

export OMP_NUM_THREADS=1
python3 src/utils/preprocess_real.py -c configs/configTransEmbed.yaml -m real
python3 src/inference/viscometer.py -c configs/configTransEmbed.yaml -m real