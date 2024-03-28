# Fine-tuning on Words and Perplexity as Features for Detecting Machine Written Text

Code made for SemEval 2024 task 8. The task deals with the identification of the authorship of the text using perplexity and text as features.

Abstract: Causation is a psychological tool of humans to understand the world and it is projected in natural language. Causation relates two events, so in order to understand the causal relation of those events and the causal reasoning of humans, the study of causality classification is required. We claim that the use of linguistic features may restrict the representation of causality, and dense vector spaces can provide a better encoding of the causal meaning of an utterance. Herein, we propose a neural network architecture only fed with word embeddings for the task of causality classification. Our results show that our claim holds, and we outperform the state-of-the-art on the AltLex corpus.

Contact person: Alberto José Gutiérrez Megías, agmegias@ujaen.es

Please use the following citation:

```
@InProceedings{martinezCamara:2017:iwcs2017,
  author    = {Alberto J. Guti{\'e}rrez, Alfonso and Eugenio},
  title     = {Fine-tuning on Words and Perplexity as Features for Detecting Machine Written Text},
  booktitle = {(to appear)},
  month     = (to appear),
  year      = {2024},
  pages     = {(to appear)}
}
```

## Data

The dataset is not available in this repository.

The dataset for the English monolingual task consists of 119,757 instances. In the multilingual task, the corpora comprise a total of 172,417 instances. This multilingual dataset consists of 77.48% English text.

Each instance includes the *text*, along with its corresponding *source* according to five categories: **Wikihow**, **Wikipedia**, **Reddit**, **Arxiv**, **Peerread**. In the multilingual task, we can find additional sources: **Bulgarian**, **Urdu**, **Indonesian** and **Chinese**. It also has a category that attributes the text to a specific large language model: **ChatGPT**, **Cohere**, **Bloomz**, **Davinci**, **Dolly**, or **Human** otherwise. The *gold label* is 1, if the text is machine-generated, and 0 otherwise.

Translated with DeepL.com (free version)

The data can be found at: https://github.com/mbzuai-nlp/SemEval2024-task

## System Description

We use the XLM-RoBERTa-Large basis as a linguistic model, and first evaluate its performance by fine-tuning the training data on it. Next, we evaluate the use of perplexity as a classification cue, and the third, which we submit to the shared task, is based on the joint use of the features resulting from the fine-tuning phase and the perplexity score of each sentence. The system used in this repository is the Multimodal System,

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

## Acknowledgements

This work has been partially supported by projects
CONSENSO (PID2021-122263OB-C21), MODERATES
(TED2021-130145B-I00), SocialTOX (PDC2022-133146-
C21) and FedDAP (PID2020-116118GA-I00) funded by MCIN/AEI/ 10.13039/501100011033 and by the “European Union NextGenerationEU/PRTR”.
