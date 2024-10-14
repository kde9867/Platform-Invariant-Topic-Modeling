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
## How to Run

To reproduce the results from the real-world data shown in the paper, follow the steps below:

### Data Setup
Replace the paths for the three collected platform data CSV files (`data/twitter_total_preprocessed.csv`, `data/reddit_total_preprocessed_cleaned.csv`, `data/youtube_preprocessed.csv`) in the code with the paths where your data is stored.

    1. Setting the path of the model codes in PITopic.ipynb.
     (i.e., The location of the two codes that upload the data required for PITopic models and organize the structure of PITopic are “/models/data.py” and “/models/model.py”, respectively, which can be set to “models” in PITopic.ipynb.)
    2. python PITopic.ipynb -- After completing the first step, you can run PITopic.ipynb to see the topics extracted by the model. 


## How to Evaluate

    1. In order to quantitatively analyze the topic words extracted by the model, we can measure the topic coherence using the code located in “models/coherence.py”. 
    The measures we used are Mutual Information, Topic Fiversity, and Topic coherence (NPMI, UCI). 

The average of NPMI and UCI was calculated for each platform separately, as well as across all platforms collectively. NPMI quantifies the frequency of word co-occurrence within a specific topic, while UCI assesses the frequency of topic co-occurrence across various documents; higher values indicate better topic coherence.
PITopic demonstrates good performance across all measures of topic coherence(see the below example result).

<img width="842" alt="evaluation-over-real-world" src="https://github.com/user-attachments/assets/14b0eb92-115b-4919-a38b-dd3c15b17191">


## Requirements

- `PyTorch == 2.4.1`
- `numpy == 1.26.4`
- `pandas == 2.2.2`
- `scikit-learn == 1.3.2`
- `gensim == 4.2.0`
