# ------------------
# Libreries --------
# ------------------
import os
import glob
import pandas as pd
import numpy as np
import torch
from transformers import Trainer, AutoTokenizer, EarlyStoppingCallback
import warnings
from dataclasses import dataclass, field
from typing import Optional
import json
from transformers.training_args import TrainingArguments
from multimodal_transformers.data import load_data
import sklearn.metrics as metrics
from transformers import AutoConfig
from multimodal_transformers.model import AutoModelWithTabular
from multimodal_transformers.model import TabularConfig

# ------------------
# Data Classes -----
# ------------------
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

@dataclass
class MultimodalDataTrainingArguments:
    """
    Arguments pertaining to how we combine tabular features
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    data_path: str = field(metadata={
                            'help': 'the path to the csv files containing the dataset. If create_folds is set to True'
                                    'then it is expected that data_path points to one csv containing the entire dataset'
                                    'to split into folds. Otherwise, data_path should be the folder containing'
                                    'train.csv, test.csv, (and val.csv if available)'
                        })
    create_folds: bool = field(default=False,
                                metadata={'help': 'Whether or not we want to create folds for '
                                                'K fold evaluation of the model'})

    num_folds: int = field(default=5,
                            metadata={'help': 'The number of folds for K fold '
                                            'evaluation of the model. Will not be used if create_folds is False'})
    validation_ratio: float = field(default=0.2,
                                    metadata={'help': 'The ratio of dataset examples to be used for validation across'
                                                    'all folds for K fold evaluation. If num_folds is 5 and '
                                                    'validation_ratio is 0.2. Then a consistent 20% of the examples will'
                                                    'be used for validation for all folds. Then the remaining 80% is used'
                                                    'for K fold split for test and train sets so 0.2*0.8=16%  of '
                                                    'all examples is used for testing and 0.8*0.8=64% of all examples'
                                                    'is used for training for each fold'}
                                    )
    num_classes: int = field(default=-1,
                            metadata={'help': 'Number of labels for classification if any'})
    column_info_path: str = field(
        default=None,
        metadata={
            'help': 'the path to the json file detailing which columns are text, categorical, numerical, and the label'
    })

    column_info: dict = field(
        default=None,
        metadata={
            'help': 'a dict referencing the text, categorical, numerical, and label columns'
                    'its keys are text_cols, num_cols, cat_cols, and label_col'
    })

    categorical_encode_type: str = field(default='none',
                                        metadata={
                                            'help': 'sklearn encoder to use for categorical data',
                                            'choices': ['ohe', 'binary', 'label', 'none']
                                        })
    numerical_transformer_method: str = field(default='yeo_johnson',
                                            metadata={
                                                'help': 'sklearn numerical transformer to preprocess numerical data',
                                                'choices': ['yeo_johnson', 'box_cox', 'quantile_normal', 'none']
                                            })
    task: str = field(default="classification",
                        metadata={
                        "help": "The downstream training task",
                        "choices": ["classification", "regression"]
                        })

    mlp_division: int = field(default=4,
                            metadata={
                                'help': 'the ratio of the number of '
                                        'hidden dims in a current layer to the next MLP layer'
                            })
    combine_feat_method: str = field(default='individual_mlps_on_cat_and_numerical_feats_then_concat',
                                    metadata={
                                        'help': 'method to combine categorical and numerical features, '
                                                'see README for all the method'
                                    })
    mlp_dropout: float = field(default=0.1,
                                metadata={
                                'help': 'dropout ratio used for MLP layers'
                                })
    numerical_bn: bool = field(default=True,
                                metadata={
                                'help': 'whether to use batchnorm on numerical features'
                                })
    use_simple_classifier: str = field(default=True,
                                        metadata={
                                        'help': 'whether to use single layer or MLP as final classifier'
                                        })
    mlp_act: str = field(default='relu',
                        metadata={
                            'help': 'the activation function to use for finetuning layers',
                            'choices': ['relu', 'prelu', 'sigmoid', 'tanh', 'linear']
                        })
    gating_beta: float = field(default=0.2,
                                metadata={
                                    'help': "the beta hyperparameters used for gating tabular data "
                                            "see https://www.aclweb.org/anthology/2020.acl-main.214.pdf"
                                })

    def __post_init__(self):
        assert self.column_info != self.column_info_path, 'provide either a path to column_info or a dictionary'
        assert 0 <= self.validation_ratio <= 1, 'validation_ratio must be between 0 and 1'
        if self.column_info is None and self.column_info_path:
            with open(self.column_info_path, 'r') as f:
                self.column_info = json.load(f)
            assert 'text_cols' in self.column_info and 'label_col' in self.column_info
            if 'cat_cols' not in self.column_info:
                self.column_info['cat_cols'] = None
                self.categorical_encode_type = 'none'
            if 'num_cols' not in self.column_info:
                self.column_info['num_cols'] = None
                self.numerical_transformer_method = 'none'
            if 'text_col_sep_token' not in self.column_info:
                self.column_info['text_col_sep_token'] = None


# ------------------
# Functions --------
# ------------------
def set_seed(seed=36):
    """Seed for reproducible experiments."""
    os.environ["SEED"] = str(seed)
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def model_init():
    model = AutoModelWithTabular.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir
    )
    return model

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(logits[0], axis=1)

    micro_precision = metrics.precision_score(y_true=labels, y_pred=predictions, average='micro')
    macro_precision = metrics.precision_score(y_true=labels, y_pred=predictions, average='macro')
    micro_recall = metrics.recall_score(y_true=labels, y_pred=predictions, average='micro')
    macro_recall = metrics.recall_score(y_true=labels, y_pred=predictions, average='macro')
    micro_f1 = metrics.f1_score(y_true=labels, y_pred=predictions, average='micro')
    macro_f1 = metrics.f1_score(y_true=labels, y_pred=predictions, average='macro')

    unique, counts = np.unique(predictions, return_counts=True)
    with open(variables['fichero_evaluacion'], 'a') as f:
        print("----------------------------", file=f)
        print("----------------------------", file=f)
        print(np.asarray((unique, counts)).T, file=f)
        print('micro_precision',micro_precision, file=f)
        print('macro_precision',macro_precision, file=f)
        print('micro_recall',micro_recall, file=f)
        print('macro_recall',macro_recall, file=f)
        print('micro_f1',micro_f1, file=f)
        print('macro_f1',macro_f1, file=f)

    return {
        "micro_precision": micro_precision,
        "macro_precision": macro_precision,
        "micro_recall": micro_recall,
        "macro_recall": macro_recall,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1
    }

def compute_objective(metrics):
    return metrics["eval_macro_f1"]


# ------------------
# Main section -----
# ------------------
if __name__=="__main__":

    set_seed(44)

    # Dataset -> [text, ppl, labels]
    # Columns -> ['text', 'label', 'model', 'source', 'id', 'ppl']
    text_cols = ['text']
    numerical_cols = ['ppl']
    label_col = ['label']
    label_list = ['human-gpt']

    column_info_dict = {
        'text_cols': text_cols,
        'num_cols': numerical_cols,
        'label_col': label_col,
        'label_list': label_list
    }

    model_args = ModelArguments(
        model_name_or_path='/mnt/beegfs/agmegias/proyectos/huggingface_models/xlm-roberta-large'
    )

    data_args = MultimodalDataTrainingArguments(
        data_path = '/mnt/beegfs/agmegias/proyectos/SEMEVAL2024/', # Path de los datos
        combine_feat_method ='individual_mlps_on_cat_and_numerical_feats_then_concat',
        column_info = column_info_dict,
        task = 'classification',
    )

    tokenizer = AutoTokenizer.from_pretrained(
        '/mnt/beegfs/agmegias/proyectos/huggingface_models/xlm-roberta-large',
        cache_dir=model_args.cache_dir,
        truncation = True,
        model_max_length=512,
        padding = True,
    )

    train_data_toolkit = pd.read_json('<PATH>', lines=True)
    test_data_toolkit = pd.read_json('<PATH>', lines=True)

    train_data_toolkit = train_data_toolkit.sample(frac=1, random_state=variables['seed'])
    test_data_toolkit = test_data_toolkit.sample(frac=1, random_state=variables['seed'])

    train_dataset = load_data(
        data_df = train_data_toolkit,
        text_cols=data_args.column_info['text_cols'],
        tokenizer=tokenizer,
        label_col=data_args.column_info['label_col'],
        label_list=data_args.column_info['label_list'],
        numerical_cols=data_args.column_info['num_cols'],
        sep_text_token_str=tokenizer.sep_token,
        categorical_encode_type='none',
    )

    test_dataset = load_data(
        data_df = test_data_toolkit,
        text_cols=data_args.column_info['text_cols'],
        tokenizer=tokenizer,
        label_col=data_args.column_info['label_col'],
        label_list=data_args.column_info['label_list'],
        numerical_cols=data_args.column_info['num_cols'],
        sep_text_token_str=tokenizer.sep_token,
        categorical_encode_type='none',
    )

    num_labels = len(np.unique(train_dataset.labels))

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    tabular_config = TabularConfig(
        num_labels=num_labels,
        numerical_feat_dim=train_dataset.numerical_feats.shape[1],
        **vars(data_args)
        )

    config.tabular_config = tabular_config
    
    training_args = TrainingArguments(
        output_dir=variables['output_dir'],
        logging_dir=variables['logging_dir'],
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        seed=44, 
        load_best_model_at_end= True,
        metric_for_best_model= 'macro_f1',
        save_total_limit=1,

        num_train_epochs= variales[ep],
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate= variales[lr], 
        weight_decay= variales[wd],
        adam_epsilon= variales[ae],
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(3)]
    )

    trainer.train()
