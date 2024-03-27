# Fine-tuning on Words and Perplexity as Features for Detecting Machine Written Text

Code made for SemEval 2024 task 8. The task deals with the identification of the authorship of the text using perplexity and text as features.

Abstract: Causation is a psychological tool of humans to understand the world and it is projected in natural language. Causation relates two events, so in order to understand the causal relation of those events and the causal reasoning of humans, the study of causality classification is required. We claim that the use of linguistic features may restrict the representation of causality, and dense vector spaces can provide a better encoding of the causal meaning of an utterance. Herein, we propose a neural network architecture only fed with word embeddings for the task of causality classification. Our results show that our claim holds, and we outperform the state-of-the-art on the AltLex corpus.

## Data

The dataset is not available in this repository.

The dataset for the monolingual English task consists of 119,757 training instances, complemented by another 5,000 evaluation instances. In the multilingual task, the corpora comprise a total of 172,417 instances, with an allocation of 4,000 instances for the evaluation phase. This multilingual dataset is composed of 77.48% English text, with Bulgarian as a secondary language. The rest of the training dataset also incorporates languages such as Chinese, Indonesian, and Urdu. In addition, the evaluation dataset includes texts in Russian, German, and Arabic.

Each instance includes the *text*, along with its corresponding *source* according to five categories: **Wikihow**, **Wikipedia**, **Reddit**, **Arxiv**, **Peerread**. In the multilingual task, we can find additional sources: **Bulgarian**, **Urdu**, **Indonesian**, and **Chinese**. Also has a category that attributes the text to a specific large language model: **ChatGPT**, **Cohere**,  **Bloomz**, **Davinci**, **Dolly**, or **Human** in another case. The *gold label* is 1, if the text is machine-generated and 0 otherwise. The dataset presents an even distribution, with cases annotated as human or machine being approximately equal in the training and development corpora.

## System Description

We use the XLM-RoBERTa-Large base as a language model, and we first assess its performance by fine-tuning the training data on it. Then, we evaluate the use of the perplexity as a classification signal, and the third one, which we submit to the shared-task, is based on the joint use of the resulting features of the fine-tuning phase and the perplexity score of each sentence.

![SemEvalv2](https://github.com/sinai-uja/SemEval-2024-Task-8-Identification-of-machine-written-text/assets/132881769/dc246629-18b1-45fd-9528-20d75eaafa27)

## Perplexity.py

*All the paths in the code are placeholders, the paths of your system must modify them.*

This code must be executed first to extract the perplexity from the text and add it to the initial dataset.

We use the python Language Model Perplexity library (LM-PPL) to calculate the perplexity. From all the large language models available to calculate the perplexity, we use GPT2.

## SemEval-Task8.py

*All the paths in the code are placeholders, the paths of your system must modify them.*

In this code, the training of the system proposed in the competition is executed. It uses text and perplexity to train a model based on XLM-RoBERTa-Large to identify the authorship of the text.

The code is structured to use an evaluation set while training and to adjust the training parameters as desired by the user.

The training parameters are commented on using the default parameters by default.
