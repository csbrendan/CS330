# Optimizing Bias Mitigation in LMs: A Study of Fine-Tuning Techniques and Augmented Data

## Paper
https://github.com/csbrendan/CS330/blob/main/paper/CS330_Project.pdf

## Poster
https://github.com/csbrendan/CS330/blob/main/poster/CS330_Project_Poster_Final.pdf

## Video
https://www.youtube.com/watch?v=PI0ejpl-R6E&t=3s

## Requirements

- WinoBias dataset
- StereoSet evaluation framework
- PyTorch


## Attribution

The code in this project was re-used, adapted and inspired from the following sources:

https://github.com/McGill-NLP/bias-bench

https://huggingface.co/datasets/wino_bias



This project utilizes the stereoset benchmark to measure bias mitigation techniques including CFD Augmentation, Masking, and In-Context Learning.

## Run Experiments
git clone https://github.com/uclanlp/corefBias.git

Combine anti-stereotype examples from the following 2 files into your train.txt 
/corefBias/WinoBias/wino/data/anti_stereotyped_type1.txt.dev

/corefBias/WinoBias/wino/data/anti_stereotyped_type2.txt.dev

git clone https://github.com/McGill-NLP/bias-bench.git

Run: python -m pip install -e .

To finetune gpt-2:

python ft_project.py

To run stereo set against that saved model:

python /bias-bench/experiments/stereoset_debias.py --bias_type gender--load_path ‘/path/to/saved/models/‘

To generate evaluation result:

python /bias-bench/experiments/stereoset_evaluation.py --predictions_dir "/bias-bench/results/stereoset_file_output_from_above_script/“ --output_file “your_evaluation_results.json"
