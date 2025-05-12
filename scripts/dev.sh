# python3 src/utils/preprocess.py -c configs/config1.yaml -m train
# python3 src/main.py -c configs/config1.yaml

python3 src/utils/preprocess.py -c configs/config2.yaml -m train
python3 src/main.py -c configs/config2.yaml

python3 src/utils/preprocess.py -c configs/config3.yaml -m train
python3 src/main.py -c configs/config3.yaml

python3 src/utils/preprocess.py -c configs/config4.yaml -m train
python3 src/main.py -c configs/config4.yaml

python3 src/utils/preprocess.py -c configs/config5.yaml -m train
python3 src/main.py -c configs/config5.yaml