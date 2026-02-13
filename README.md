## Title
**Learning to Reason for Multi-Step Retrieval of Personal Context in Personalized Question Answering**

## Abstract
Personalization in Question Answering (QA) requires answers that are both accurate and aligned with users’ background, preferences, and historical context. Existing state-of-the-art methods primarily rely on retrieval-augmented generation (RAG) solutions that construct personal context by retrieving relevant items from the user’s profile. Existing methods use the user’s query directly to retrieve personal documents and such strategies often lead to surfacelevel personalization. We propose PR² (Personalized Retrieval-Augmented Reasoning), a reinforcement learning framework that integrates reasoning and retrieval from personal context for personalization. PR² learns adaptive retrieval-reasoning policies, determining when to retrieve, what evidence to retrieve from user profiles, and how to incorporate it into intermediate reasoning steps. By optimizing multi-turn reasoning trajectories under a personalized reward function, the framework reinforces reasoning paths that better align with user-specific preferences and contextual signals reflected by the reward model. Extensive experiments on the LaMP-QA benchmark using three LLMs show that PR² consistently outperforms strong baselines, achieving an average relative improvement of 8.8%-12% in personalized QA.


## Installing requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation
You need to download the [LaMP-QA](https://github.com/LaMP-Benchmark/LaMP-QA?tab=readme-ov-file#downloading-the-dataset) dataset. For this purpose, you can use the following code:

```bash
python download.py \
    --dataset_save_directory /path/to/download_directory \
    --cache_dir /path/to/cache_directory
```

## Retrieval from User Profile

Next, in order to prepare the RAG personalization on the client-side:

```bash
python retrieval.py \
    --input_dataset_addr /*address to the dataset file*/ \
    --output_dataset_addr /*address to where the dataset with sorted profile for each user should be saved*/ \
    --model_name "facebook/contriever-msmarco" \
    --batch_size 16
```

## Running PR²

```bash
python PR2.py \
    --questions_address /path/to/prepared_dataset.jsonl \
    --output_address /path/to/output_directory \
    --model <qwen|gemma> \
    --temperature <temp> \
    --judge-model <judge> \
    --split test \
    --subsets ALL \
    --max-model-len <max_len>
```

## Evaluation

Please use the evaluation script provided by the [LaMP-QA benchmark](https://github.com/LaMP-Benchmark/LaMP-QA?tab=readme-ov-file#evaluating-the-generated-responses) to evaluate generated responses.


