# multilayer-perceptron

## Install requirements
`bash install_requirements.sh multilayer-perceptron`

then

`source multilayer-perceptron/bin/activate`

## Split
`python split.py datasets/data.csv`

## Training
`python train.py`

## Predict
`python predict.py -d ./datasets/validation.csv -m ./saved_model.npy`