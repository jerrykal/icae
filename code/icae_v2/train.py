import transformers
from datasets import load_from_disk
from peft import (
    LoraConfig,
)

from modeling_icae_multi_span import (
    ICAE,
    DataArguments,
    ModelArguments,
    TrainingArguments,
)
from training_utils import (
    DataCollatorForDynamicPadding,
    train_model,
)


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print(model_args)
    print(data_args)

    training_args.gradient_checkpointing_kwargs = {
        "use_reentrant": False
    }  # manually add this argument in the code

    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # check model_args.mem_size and min_tokens_for_lm
    assert (
        training_args.fixed_mem_size & (training_args.fixed_mem_size - 1)
    ) == 0, "training_args.fixed_mem_size must be a power of 2"
    assert (
        training_args.leave_tokens_for_lm <= training_args.min_tokens_for_lm
    ), "leave_tokens_for_lm should be fewer than min_tokens_for_lm"

    print("Loading dataset...")

    model = ICAE(model_args, training_args, lora_config)

    train_dataset = load_from_disk(data_args.train_data[0])

    data_collator = DataCollatorForDynamicPadding(model.pad_token_id)
    train_model(model, train_dataset, None, training_args, data_collator)


main()
