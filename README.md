# MCDN
A Multi-level Neural Network for Implicit Causality Detection in Web Texts

Under review. [Public Version](https://arxiv.org/abs/1908.07822)

## Prerequisites

- pytorch >= 0.4
- nltk
- gensim
- sklearn

## Dataset

[altlex](https://github.com/chridey/altlex)

## Preprocessing

Preprocess the trainset:

```
python torch_run.py run --prepare=True --build=True
```

## Train

```
python torch_run.py run --train=True
```

## Test

```
python torch_run.py run  --evaluate=True
```

## other details is coming soon...