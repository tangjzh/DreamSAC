# iVideoGPT: Interactive VideoGPTs are Scalable World Models

[[Website]](https://thuml.github.io/iVideoGPT/) [[Paper]](https://arxiv.org/abs/2405.15223) [[Model]](https://huggingface.co/thuml/ivideogpt-oxe-64-act-free)

This repo provides official code and checkpoints for iVideoGPT, a generic and efficient world model architecture that has been pre-trained on millions of human and robotic manipulation trajectories. 

![architecture](assets/architecture.png)

## News

- 🚩 **2024.08.31**: Training code is released (Work in progress 🚧 and please stay tuned!)
- 🚩 **2024.05.31**: Project website with video samples is released.
- 🚩 **2024.05.30**: Model pre-trained on Open X-Embodiment and inference code are released.
- 🚩 **2024.05.27**: Our paper is released on [arXiv](https://arxiv.org/abs/2405.15223).

## Installation

```bash
conda create -n ivideogpt python==3.9
conda activate ivideogpt
pip install -r requirements.txt
```

## Models

At the moment we provide the following models:

| Model | Resolution | Action | Tokenizer Size | Transformer Size |
| ---- | ---- | ---- | ---- | ---- |
| [ivideogpt-oxe-64-act-free](https://huggingface.co/thuml/ivideogpt-oxe-64-act-free) | 64x64 | No |  114M   |  138M    |

If no network connection to Hugging Face, you can manually download from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/ef7d94c798504587a95e/).

## Inference Examples

### Action-free Video Prediction on Open X-Embodiment

```bash
python inference/predict.py --pretrained_model_name_or_path "thuml/ivideogpt-oxe-64-act-free" --input_path inference/samples/fractal_sample.npz --dataset_name fractal20220817_data
```

To try more samples, download the dataset from the [Open X-Embodiment Dataset](https://robotics-transformer-x.github.io/) and extract single episodes as follows:

```bash
python oxe_data_converter.py --dataset_name {dataset_name, e.g. bridge} --input_path {path to OXE} --output_path samples --max_num_episodes 10
```

## Training Video Prediction

### Pretrained Models

To finetune our [pretrained iVideoGPT](https://huggingface.co/thuml/ivideogpt-oxe-64-act-free), download it into `pretrained_models/ivideogpt-oxe-64-act-free`.

To evaluate the FVD metric, download [pretrained I3D model](https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1) into `pretrained_models/i3d/i3d_torchscript.pt`.

### Data Preprocessing

**BAIR Robot Pushing**: Download the [dataset](http://rail.eecs.berkeley.edu/datasets/bair_robot_pushing_dataset_v0.tar) and preprocess with the following script:

```bash
wget http://rail.eecs.berkeley.edu/datasets/bair_robot_pushing_dataset_v0.tar -P .
tar -xvf ./bair_robot_pushing_dataset_v0.tar -C .

python datasets/preprocess_bair.py --input_path bair_robot_pushing_dataset_v0/softmotion30_44k --save_path bair_preprocessed
```

Then modify the saved paths (e.g. `bair_preprocessed/train` and `bair_preprocessed/test`) in `DATASET.yaml`.

### Finetuning Tokenizer

```bash
accelerate launch train_tokenizer.py \
    --exp_name bair_tokenizer_ft --output_dir log_vqgan --seed 0 --mixed_precision bf16 \
    --model_type ctx_vqgan \
    --train_batch_size 16 --gradient_accumulation_steps 1 --disc_start 1000005 \
    --oxe_data_mixes_type bair --resolution 64 --dataloader_num_workers 16 \
    --rand_select --video_stepsize 1 --segment_horizon 16 --segment_length 8 --context_length 1 \
    --pretrained_model_name_or_path pretrained_models/ivideogpt-oxe-64-act-free/tokenizer
```

### Finetuning Transformer

For action-conditioned video prediction, run the following:

```bash
accelerate launch train_gpt.py \
    --exp_name bair_llama_ft --output_dir log_trm --seed 0 --mixed_precision bf16 \
    --vqgan_type ctx_vqgan \
    --pretrained_model_name_or_path {log directory of finetuned tokenizer}/unwrapped_model \
    --config_name configs/llama/config.json --load_internal_llm --action_conditioned --action_dim 4 \
    --pretrained_transformer_path pretrained_models/ivideogpt-oxe-64-act-free/transformer \
    --per_device_train_batch_size 16 --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 --lr_scheduler_type cosine \
    --oxe_data_mixes_type bair --resolution 64 --dataloader_num_workers 16 \
    --video_stepsize 1 --segment_length 16 --context_length 1 \
    --use_eval_dataset --use_fvd --use_frame_metrics \
    --weight_decay 0.01 --llama_attn_drop 0.1 --embed_no_wd
```

For action-free video prediction, remove `--load_internal_llm --action_conditioned`.

## Training Visual Model-based RL

### Preparation

Install the Metaworld version we used:

```bash
pip install git+https://github.com/Farama-Foundation/Metaworld.git@83ac03ca3207c0060112bfc101393ca794ebf1bd
```

Modify paths in `mbrl/cfgs/mbpo_config.yaml` to your own paths (currently only support absolute paths).

### MBRL with iVideoGPT

```bash
python mbrl/train_metaworld_mbpo.py task=plate_slide num_train_frames=100002 demo=true
```

## Showcases

![showcase](assets/showcase.png)

## Citation

If you find this project useful, please cite our paper as:

```
@article{wu2024ivideogpt,
    title={iVideoGPT: Interactive VideoGPTs are Scalable World Models}, 
    author={Jialong Wu and Shaofeng Yin and Ningya Feng and Xu He and Dong Li and Jianye Hao and Mingsheng Long},
    journal={arXiv preprint arXiv:2405.15223},
    year={2024},
}
```

## Contact

If you have any question, please contact wujialong0229@gmail.com.

## Acknowledgement

Our codebase is heavily built upon [huggingface/diffusers](https://github.com/huggingface/diffusers) and [facebookresearch/drqv2](https://github.com/facebookresearch/drqv2).
