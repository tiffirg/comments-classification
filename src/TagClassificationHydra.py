import re
import torch
import polars as pl
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
import warnings
from autogluon.tabular import TabularPredictor
import seaborn as sns
from typing import Any
from functools import partial
from datasets import load_dataset, Dataset
import optuna
from optuna.samplers import TPESampler
from nltk.corpus import stopwords
from string import punctuation as PUNCT
from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    accuracy_score,
    multilabel_confusion_matrix,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    hamming_loss,
    precision_score,
    recall_score,
)
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from hydra.core.config_store import ConfigStore
from data_types import (
    RubertModel,
    RubertPreprocessing,
    RubertTraining,
    MainConfig,
    SecretsConfig
)
warnings.filterwarnings("ignore")

cs = ConfigStore.instance()
cs.store(group="preprocessing", name="rubert", node=RubertPreprocessing)
cs.store(group="model", name="rubert", node=RubertModel)
cs.store(group="training", name="rubert", node=RubertTraining)
cs.store(group="secrets", name="secrets", node=SecretsConfig)
cs.store(name="config", node=MainConfig)


def remove_sub_tags(tags: str):
    split = tags.split(sep=" ")
    new_tag = [x[:-1] if x[-1].isdigit() else x for x in split]
    return " ".join(new_tag)


def multi_label_metrics(
    predictions: np.ndarray, labels: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Compute mltilabel metrics.

    Args:
        predictions (np.ndarray): logits array
        labels (np.ndarray): labels array
        threshold (float, optional): activation threshold. Defaults to 0.5.

    Returns:
        dict[str, float]: metrics dict
    """
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    f1_micro_average = f1_score(y_true=labels, y_pred=y_pred, average="micro")
    roc_auc = roc_auc_score(labels, y_pred, average="micro")
    accuracy = accuracy_score(labels, y_pred)
    metrics = {"f1": f1_micro_average, "roc_auc": roc_auc, "accuracy": accuracy}
    return metrics


def compute_metrics(p: EvalPrediction) -> dict[str, float]:
    """Metrics computation wrapper.

    Args:
        p (EvalPrediction): hf model output

    Returns:
        dict[str, float]: metrics dict
    """
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(predictions=preds, labels=p.label_ids)
    return result


def make_training_pipeline(
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    model_conf: dict[str, Any],
    train_conf: dict[str, Any],
    num_labels,
    id2label: dict[int, str],
    label2id: dict[str, int],
    model_name: str
) -> Trainer:
    """Training process wrapper.

    Args:
        exp_name (str): name of the local folder
        for saving model checkpoints.
        tokenizer (AutoTokenizer): model tokenizer
        train_dataset (Dataset): train dataset split
        eval_dataset (Dataset): test dataset split
        batch_size (int, optional): number of samples
        in sigle batch. Defaults to 32.
        lr (float, optional): model's learning rate. Defaults to 2e-5.
        epochs_num (int, optional):
        number of training iterations. Defaults to 20.

    Returns:
        Trainer: hf training pipeline abstraction class.
    """
    args = instantiate(train_conf)

    model = instantiate(
        model_conf, num_labels=num_labels, id2label=id2label, label2id=label2id
    )
    
    model = instantiate(
        model_conf, num_labels=len(id2label), id2label=id2label, label2id=label2id
    )
    
    if model_name is not None:
        model.load_state_dict(torch.load(model_name))

    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    return trainer


@hydra.main(version_base=None, config_name="config")
def main(cfg: DictConfig):
    dataset = pl.read_csv("preprocessing.csv")

    dataset = dataset.with_columns(
        (pl.col("Тег").apply(lambda x: " ".join(re.findall(
            r"[A-Z]{1,2}\d|LMS", x)))).alias("corrected_tag")
    )

    null_filter = (
        (pl.col("corrected_tag").eq(""))
    )

    dataset = dataset.filter(~null_filter)
    dataset = dataset.filter(~(pl.col("Комментарий").is_null()))
    dataset = dataset.with_columns(
        pl.col("corrected_tag")
        .str.replace_all(r"VC4|VP4|VC5|S4|T4|H4|EA1", "")
        .str.strip()
        .str.replace(r"\s\s+", " ")
        .str.replace(r"GH3", "H3")
        .str.replace(r"HH3", "H3")
        .str.replace(r"BP3", "VP3")
        .str.replace(r"V3", "VC3")
        .str.replace(r"V2", "VP2"))

    dataset = dataset.filter(~(pl.col("corrected_tag").eq("")))
    dataset["corrected_tag"].str.split(
        by=" ").explode().value_counts(sort=True)
    dataset = dataset.filter(~pl.col("corrected_tag").str.contains("E2"))

    dataset = dataset.with_columns(
        pl.col("corrected_tag").apply(remove_sub_tags)
    )

    dataset["corrected_tag"].str.split(
        by=" ").explode().value_counts(sort=True)
    target = dataset["corrected_tag"].str.split(
        by=" ").explode().unique().sort().to_list()
    target = dict(zip(target, range(len(target))))
    reverse_target = {v: k for k, v in target.items()}

    def vectorize(tags: str) -> list[float]:
        """Turn str with tags into list with digit labels.

        Args:
            tags (str): tag text representation.

        Returns:
            list[float]: numeric labels.
        """
        split = tags.split(sep=" ")
        res = np.zeros(len(target))
        for x in split:
            res[target[x]] = 1
        return res.tolist()

    dataset = dataset.with_columns(
        pl.col("corrected_tag").apply(vectorize).alias("labels"))
    clear_dataset = dataset.select(
        pl.col("Комментарий"),
        pl.col("Направление"),
        pl.col("Факультет"),
        pl.col("Оценка"),
        pl.col("Neutral"),
        pl.col("Positive"),
        pl.col("Negative"),
        pl.col("Exclamations"),
        pl.col("have_code"),
        pl.col("Neutral_NLP"),
        pl.col("Positive_NLP"),
        pl.col("Negative_NLP"),
        pl.col("Speech_NLP"),
        pl.col("corrected_tag"),
        pl.col("labels"),
        pl.col("corrected_tag").str.split(by=" ").alias("temp"))
    clear_dataset = clear_dataset.explode(columns=["temp"])
    train_df, test_df = train_test_split(
        clear_dataset,
        test_size=cfg["test_size"],
        random_state=3317,
        stratify=clear_dataset["temp"])

    train_df = train_df.drop(columns=["corrected_tag", "temp"])
    test_df = test_df.drop(columns=["corrected_tag", "temp"])

    train_df = train_df.rename({"Комментарий": "text"})
    test_df = test_df.rename({"Комментарий": "text"})
    
    train_dataset = Dataset.from_pandas(train_df.to_pandas(), split="train")
    test_dataset = Dataset.from_pandas(test_df.to_pandas(), split="test")
    
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["preprocessing"]["tokenizer_name"], token=cfg["secrets"]["hf_token"]
    )
    
    def preprocess_data(sample: dict[str, Any]) -> dict[str, Any]:
        """Encode input text into sequence of tokens.
        Also add corresponding labels.

        Args:
            sample (dict[str, Any]): raw input text.

        Returns:
            dict[str, Any]: transformed sample with tokenized text and labels.
        """
        text = sample["text"]
        encoding = tokenizer(
            text,
            padding=cfg["preprocessing"]["padding"],
            truncation=True,
            max_length=cfg["preprocessing"]["max_length"],
        )
        encoding["labels"] = sample["labels"]
        return encoding
    
    encoded_train = train_dataset.map(
    preprocess_data, batched=True, remove_columns=train_dataset.column_names)
    encoded_test = test_dataset.map(
        preprocess_data, batched=True, remove_columns=test_dataset.column_names)
    encoded_train.set_format("torch")
    encoded_test.set_format("torch")
    trainer = make_training_pipeline(tokenizer,
                                     encoded_train,
                                     encoded_test,
                                     cfg["model"], cfg["training"],
                                     num_labels=len(target),
                                     id2label=target,
                                     label2id=reverse_target)
    trainer.train()
    trainer.evaluate()
    train_preds = trainer.predict(encoded_train)
    test_preds = trainer.predict(encoded_test)
    print(compute_metrics(test_preds))
    
    directions = pd.get_dummies(train_df.to_pandas()["Направление"])
    departments = pd.get_dummies(train_df.to_pandas()["Факультет"])
    meta_dataset_train = train_df.select(pl.exclude("Направление", "Факультет")).to_pandas()
    meta_dataset_train = pd.concat([meta_dataset_train, directions, departments, pd.DataFrame(train_preds.predictions)], axis=1)
    meta_dataset_train = meta_dataset_train.drop(columns=["text"])

    meta_dataset_test = test_df.select(pl.exclude("Направление", "Факультет")).to_pandas()
    meta_dataset_test = pd.concat([meta_dataset_test, directions, departments, pd.DataFrame(test_preds.predictions)], axis=1)
    meta_dataset_test = meta_dataset_test.drop(columns=["text"])

    meta_dataset_test = meta_dataset_test.dropna(subset=["labels"])
    
    X_train, y_train = meta_dataset_train.drop('labels', axis=1), np.array(meta_dataset_train["labels"].to_list())
    X_test, y_test = meta_dataset_test.drop('labels', axis=1), np.array(meta_dataset_test["labels"].to_list())
    train_pool = Pool(X_train, y_train)
    test_pool = Pool(X_test, y_test)
    
    def objective(trial):
        model = CatBoostClassifier(
            iterations=trial.suggest_int("iterations", 500, 2000),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
            depth=trial.suggest_int("depth", 4, 10),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-8, 100.0, log=True),
            bootstrap_type=trial.suggest_categorical("bootstrap_type", ["Bayesian"]),
            random_strength=trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 10.0),
            od_type=trial.suggest_categorical("od_type", ["IncToDec", "Iter"]),
            od_wait=trial.suggest_int("od_wait", 10, 50),
            verbose=False,
            task_type="GPU",
            devices='0',
            loss_function = trial.suggest_categorical("loss_function", ["MultiCrossEntropy", "MultiLogloss"]))
        model.fit(train_pool, eval_set=test_pool)
        y_pred = model.predict(test_pool)
        return hamming_loss(y_test, y_pred)
    
    sampler = TPESampler(seed=1337)
    study = optuna.create_study(study_name="catboost", direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=cfg["num_trials"])
    
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        
    model = CatBoostClassifier(**trial.params, verbose=True)
    model.fit(train_pool, eval_set=test_pool)
    pred_labels = model.predict(test_pool)
    
    cr = classification_report(np.array(meta_dataset_test["labels"].to_list()), pred_labels, output_dict=True)
    cr = pd.DataFrame(cr).T
    print(cr)


if __name__ == "__main__":
    main()
