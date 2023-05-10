# cpu_test

## run openvino test

prepare environment
```
conda create -y -n vino_test python=3.10
conda activate vino_test
pip install -r requirements.txt
```

run openvino performance test

```
python script/openvino_test.py
```