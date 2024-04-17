PYTHON_BUFFER=10
exp_path="$(python3 writeHyperparam.py)"
python3 train.py $exp_path