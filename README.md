# softtriple_pytorch
Unofficial implementation of `SoftTriple Loss: Deep Metric Learning Without Triplet Sampling`

## how to install

pip install git+https://github.com/nkennek/softtriple_pytorch.git#egg=softtriple_pytorch

## how to run experiment

1. clone this repository and cd
2. run `pipenv install --dev && pipenv run python -m bin.cub2011_experiment`

Note: I have not succssfully reproduce R@1 of 60.1% with CUB-2001 yet.
Any further trials are welcome.