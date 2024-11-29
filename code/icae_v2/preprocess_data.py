import transformers
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
    prepare_train_data,
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

    memory_size = training_args.fixed_mem_size

    print("Loading dataset...")

    # Model's here is only used for the tokenized_function
    model = ICAE(model_args, training_args, lora_config)
    MEM_TOKENS = list(range(model.vocab_size, model.vocab_size + memory_size))

    train_dataset = prepare_train_data(
        model, MEM_TOKENS, training_args.lm_ratio, data_args.train_data
    )

    # Save train_dataset
    train_dataset.save_to_disk(training_args.output_dir)


main()
