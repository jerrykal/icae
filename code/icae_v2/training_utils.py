import itertools
import logging
import math
import os
import random
import re

import numpy as np
import torch
from datasets import (
    Dataset,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from transformers import (
    Trainer,
)
from transformers.trainer_utils import get_last_checkpoint

from chat_utils import apply_chat_template

logger = logging.getLogger(__name__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, train_dataset, eval_dataset, training_args, data_collator=None):
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            print(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    if (
        max(
            training_args.per_device_train_batch_size,
            training_args.per_device_eval_batch_size,
        )
        == 1
    ):
        data_collator = None

    # print training_args at local_rank 0
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if local_rank == 0:
        print(training_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=model.tokenizer,
    )

    checkpoint = None

    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    print(f"Loaded from the checkpoint: {checkpoint}")

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    # metrics = trainer.evaluate()
    # trainer.log_metrics("eval", metrics)
    # trainer.save_metrics("eval", metrics)


def text_extraction(input_ids, length, lm_ratio=0.0):
    input_len = len(input_ids)
    assert input_len >= 1, f"Error: invalid input length ({input_len})"

    # ae
    if random.random() >= lm_ratio:
        if input_len <= length:  # if shorter, keep the complete text
            return input_ids, []
        else:
            last_start = input_len - length
            random_start = random.randint(0, last_start)
            return input_ids[random_start : random_start + length], []

    # lm
    if input_len <= length:
        r = random.randint(1024, input_len - 1)
        return input_ids[: r + 1], input_ids[r + 1 :]
    else:
        last_start = input_len - length
        random_start = random.randint(0, last_start)
        return input_ids[random_start : random_start + length], input_ids[
            random_start + length :
        ]


def compute_num_segments(total_length, mem_size, mean_compression_rate):
    assert total_length > 0
    num_segments = math.ceil(total_length / (mem_size * mean_compression_rate))
    return num_segments


def pretrain_tokenize_function(
    examples, tokenizer, training_args, ae_token_id, lm_token_id, eos_id, mem, lm_ratio
):
    if "text" in examples.keys():
        text_output = tokenizer(
            examples["text"],
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )
    elif "input_ids" in examples.keys():
        text_output = examples
    else:
        raise ValueError("Found neither 'text' nor 'input_ids' in the training data!")

    text_output["prompt_answer_ids"] = []
    text_output["labels"] = []

    max_len = training_args.model_max_length  # heuristic

    for idx in range(len(text_output["input_ids"])):
        ae = True
        a, b = text_extraction(
            text_output["input_ids"][idx], max_len, lm_ratio=lm_ratio
        )
        num_segments = compute_num_segments(
            len(a), training_args.fixed_mem_size, training_args.mean_compression_rate
        )
        total_mem_length = num_segments * training_args.fixed_mem_size

        if (
            len(b) > training_args.min_tokens_for_lm
        ):  # avoid too few tokens for lm, which is a waste of computing
            ae = False
            b = b[:max_len]

        text_output["input_ids"][idx] = a

        # decoder part: note that in v2, we add mem_tokens to the prompt_ids for easy implementation; which is different from v1 implementation where mem tokens are not in the prompt_ids
        if ae:  # autoencoding objective
            prompt_ids = [mem[0]] * total_mem_length + [ae_token_id]
            answer_ids = a + [eos_id]  # if ae, eos token
        else:  # lm objective
            prompt_ids = [mem[0]] * total_mem_length
            if training_args.add_special_token_for_lm:
                prompt_ids += [lm_token_id]
            answer_ids = b  # if lm, no eos token

        text_output["prompt_answer_ids"].append(prompt_ids + answer_ids)
        if ae:
            labels = [-100] * len(prompt_ids) + answer_ids
        else:
            labels = (
                [-100] * len(prompt_ids)
                + [-100] * training_args.leave_tokens_for_lm
                + answer_ids[training_args.leave_tokens_for_lm :]
            )  # no loss for leave_tokens_for_lm
        text_output["labels"].append(labels)
        assert len(text_output["prompt_answer_ids"][-1]) == len(labels)

    return text_output


def instruct_ft_tokenize_function(
    examples, tokenizer, training_args, ft_token_id, eos_id, mem
):
    text_output = {
        "input_ids": [],
        "prompt_answer_ids": [],
        "labels": [],
    }

    maxlen = training_args.model_max_length
    minlen = training_args.model_min_length

    for conversation in examples["conversations"]:
        # Skip the first one if it is not from user
        if conversation[0]["role"] != "user":
            conversation = conversation[1:]

        encoded = apply_chat_template(
            training_args.chat_template,
            conversation,
            tokenizer=tokenizer,
            return_labels=True,
        ).encoded

        # Make sure the conversation is between minlen and maxlen
        if minlen is not None and len(encoded["input_ids"]) < minlen:
            continue
        if maxlen is not None and len(encoded["input_ids"]) > maxlen:
            continue

        segment_lengths = [
            len(list(segment))
            for _, segment in itertools.groupby(np.array(encoded["labels"]) == -100)
        ]
        assert sum(segment_lengths) == len(encoded["labels"])

        curr_pos = 0
        input_ids = []
        last_input_id = []

        for i, segment_length in enumerate(segment_lengths):
            segment_input_ids = encoded["input_ids"][
                curr_pos : curr_pos + segment_length
            ]

            if i % 2 == 0 and len(input_ids) > 0:
                num_segments = compute_num_segments(
                    len(input_ids),
                    training_args.fixed_mem_size,
                    training_args.mean_compression_rate,
                )
                total_mem_length = num_segments * training_args.fixed_mem_size

                prompt_ids = [mem[0]] * total_mem_length + [ft_token_id] + last_input_id
                answer_ids = segment_input_ids + [eos_id]

                labels = [-100] * len(prompt_ids) + answer_ids

                text_output["input_ids"].append(input_ids.copy())
                text_output["prompt_answer_ids"].append(prompt_ids + answer_ids)
                text_output["labels"].append(labels)

                assert len(text_output["prompt_answer_ids"][-1]) == len(labels)

            input_ids.extend(last_input_id)
            last_input_id = segment_input_ids
            curr_pos += segment_length

    return text_output


class DataCollatorForDynamicPadding:
    def __init__(self, pad_token_id, pad_to_multiple_of=None):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, examples):
        input_ids = [
            torch.tensor(example["input_ids"], dtype=torch.long) for example in examples
        ]
        labels = [
            torch.tensor(example["labels"], dtype=torch.long) for example in examples
        ]
        prompt_answer_ids = [
            torch.tensor(example["prompt_answer_ids"], dtype=torch.long)
            for example in examples
        ]
        input_ids = self.dynamic_padding(input_ids, fill_value=self.pad_token_id)
        prompt_answer_ids = self.dynamic_padding(
            prompt_answer_ids, fill_value=self.pad_token_id
        )
        labels = self.dynamic_padding(labels)
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "prompt_answer_ids": prompt_answer_ids,
        }
        return batch

    def dynamic_padding(self, sequences, fill_value=-100):
        max_length = max(len(x) for x in sequences)
        if self.pad_to_multiple_of:
            max_length = (
                (max_length - 1) // self.pad_to_multiple_of + 1
            ) * self.pad_to_multiple_of
        padded_sequences = torch.full(
            (len(sequences), max_length), fill_value, dtype=torch.long
        )
        for i, seq in enumerate(sequences):
            padded_sequences[i, : len(seq)] = seq
        return padded_sequences


def prepare_train_data(
    model,
    mem,
    lm_ratio,
    data_files: list | str = None,
    seed: int = 42,
    cache_dir: str | None = None,
    load_from_cache_file: bool | None = None,
) -> Dataset:
    if data_files is None:
        return None

    if isinstance(data_files, list):
        logger.info(f"Loading training data from {data_files}...")
    elif isinstance(data_files, str):
        logger.info(f"Loading training data from {data_files}...")
        data_files = [data_files]
    else:
        raise ValueError(f"Invalid training data {data_files}!")

    data_2_num_sample = {}
    for data_file in data_files:
        match = re.search("\[(\d*)\]", data_file)
        if match:
            max_sample_num = int(match.group(1))
            data_file = re.sub("\[(\d*)\]", "", data_file)
        else:
            max_sample_num = None
        data_2_num_sample[data_file] = max_sample_num

    random.seed(seed)

    train_datasets = []
    for data_file, max_sample_num in data_2_num_sample.items():
        print(f"Processing {data_file}...")
        if os.path.isdir(data_file) and os.path.exists(
            os.path.join(data_file, "dataset_info.json")
        ):
            # the dataset may be save_to_disk in advance
            dataset = load_from_disk(data_file)
            column_names = [
                col
                for col in dataset.column_names
                if col != "input_ids" and col != "labels"
            ]
            map_fn = pretrain_tokenize_function
            fn_kwargs = {
                "tokenizer": model.tokenizer,
                "training_args": model.training_args,
                "ae_token_id": model.ae_token_id,
                "lm_token_id": model.lm_token_id,
                "eos_id": model.eos_id,
                "mem": mem,
                "lm_ratio": lm_ratio,
            }

        else:
            # the dataset is a json file
            dataset = load_dataset(
                "json", data_files=data_file, split="train", cache_dir=cache_dir
            )

            column_names = dataset.column_names
            if "text" in column_names:
                map_fn = pretrain_tokenize_function
                fn_kwargs = {
                    "tokenizer": model.tokenizer,
                    "training_args": model.training_args,
                    "ae_token_id": model.ae_token_id,
                    "lm_token_id": model.lm_token_id,
                    "eos_id": model.eos_id,
                    "mem": mem,
                    "lm_ratio": lm_ratio,
                }
            # TODO: add instruction tuning preprocessing later
            elif "conversations" in column_names:
                map_fn = instruct_ft_tokenize_function
                fn_kwargs = {
                    "tokenizer": model.tokenizer,
                    "training_args": model.training_args,
                    "ft_token_id": model.ft_token_id,
                    "eos_id": model.eos_id,
                    "mem": mem,
                }
            else:
                raise ValueError(
                    "Found neither 'text' nor 'conversations' in the training data!"
                )

        dataset = dataset.map(
            map_fn,
            batched=True,
            num_proc=32,
            remove_columns=column_names,
            batch_size=32,
            load_from_cache_file=load_from_cache_file,
            fn_kwargs=fn_kwargs,
        )

        if max_sample_num is not None and len(dataset) > max_sample_num:
            dataset = dataset.train_test_split(max_sample_num, seed=seed)["test"]

        # index column is useless in training
        if "index" in dataset.column_names:
            dataset = dataset.remove_columns(["index"])

        print(f"Dataset length: {len(dataset)}")

        train_datasets.append(dataset)

    dataset = concatenate_datasets(train_datasets)

    return dataset
