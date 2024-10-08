# Platform-Invariant Topic Modeling via Contrastive Learning to Mitigate Platform-Induced Bias


Implementation of Platform-Invariant Topic Modeling. 

This study enhances the performance of topic models by developing a platform-invariant contrastive learning algorithm and eliminating platform-specific jargon sets to minimize the unique influence of each platform.

## Data

For this study, data was collected directly from three platforms: X, Reddit, and YouTube, respectively. You can find three datasets in the sub-directory named `data`. Each file is raw data collected from three platforms (X, Reddit, YouTube) using the keyword “ChatGPT”. You can utilize this data when implementing the PITopic.

<img src="https://github.com/user-attachments/assets/fb061bce-9e87-4193-be6e-8802719c2b91" alt="Data" width="500"/>


## Usage

`PITopic.ipynb` is the main notebook for training and evaluating the model. This code contains a platform-invariant contrastive learning algorithm and removes platform-specific jargon word sets.

![PITopic_main_model](https://github.com/user-attachments/assets/07b31cfe-39c8-40b3-8fc4-eac67844cbc0)


This research enhances the potential for robust social analysis across diverse platforms by contributing to more accurate and unbiased topic discovery.

## Requirements

- `PyTorch == 2.4.1`
- `numpy == 1.26.4`
- `pandas == 2.2.2`
- `scikit-learn == 1.3.2`
- `gensim == 4.2.0`
