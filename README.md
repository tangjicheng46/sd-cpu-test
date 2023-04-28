# cpu_test

## environment

```
conda create -n cpu_test python=3.10
pip install -r requirements.txt
```

## onnx export

```
optimum-cli export onnx --model runwayml/stable-diffusion-v1-5 sd_v15_onnx/
```