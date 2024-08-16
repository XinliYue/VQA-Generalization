## Revisiting Video Quality Assessment from the Perspective of Generalization

Code for AAAI 2025 submission paper "Revisiting Video Quality Assessment from the Perspective of Generalization".

## Environment

- Python (3.11.5)
- Pytorch (2.1.2)
- torchvision (0.16.2)
- CUDA (12.0)

## Content

- ```./fastvqa```: Code for FAST-VQA-AWP.
- ```./sama```: Code for SAMA-AWP.
- ```./simplevqa```: Code for SimpleVQA-AWP.

## Run

- FAST-VQA-wd

```bash
cd ./fastvqa
CUDA_VISIBLE_DEVICES='0' python train.py --wd 0.0005
```

- FAST-VQA-AuA

```bash
cd ./fastvqa
CUDA_VISIBLE_DEVICES='0' python train.py --aua
```

- FAST-VQA-RWP

```bash
cd ./fastvqa
CUDA_VISIBLE_DEVICES='0' python train.py --rwp --gamma 0.0001
```

- FAST-VQA-AWP

```bash
cd ./fastvqa
CUDA_VISIBLE_DEVICES='0' python train.py --awp --gamma 0.0001
```

- SimpleVQA-AWP

```bash
cd ./simplevqa
CUDA_VISIBLE_DEVICES='0' python train_LSVQ.py --awp --gamma 0.0001
```

- SAMA-AWP

```bash
cd ./sama
CUDA_VISIBLE_DEVICES='0' python train.py --awp --gamma 0.0001
```

## Reference Code

[1] SimpleVQA: https://github.com/sunwei925/SimpleVQA

[2] FAST-VQA: https://github.com/VQAssessment/FAST-VQA-and-FasterVQA

[3] SAMA: https://github.com/Sissuire/SAMA/tree/main