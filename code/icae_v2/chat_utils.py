# Based on https://github.com/FlagOpen/FlagEmbedding/blob/master/research/Long_LLM/activation_beacon/src/chat.py

import dataclasses
from copy import deepcopy

import numpy as np
from fastchat.conversation import get_conv_template
from transformers.tokenization_utils import BatchEncoding, PreTrainedTokenizer

# turn_sep: separator for different conversation turns (one turn consists of an utterance from user and a response from assistant)
# role_sep: separator for different roles within each turn
# begin_of_text_len: the number of tokens in the beginning of the entire sequence, usually the length of the bos string
# turn_seq_left_offset: the number of tokens to offset in the beginning of each turn, these tokens should be masked
CHAT_TEMPLATE_CONFIG = {
    "mistral": {
        "turn_sep": "</s>",
        "role_sep": " [/INST]",
        "begin_of_text_len": 1,
        "turn_seq_left_offset": 0,
    },
    "llama-2": {
        "turn_sep": " </s><s>",
        "role_sep": " [/INST]",
        "begin_of_text_len": 1,
        "turn_seq_left_offset": -1,
    },
    "llama-3": {
        "turn_sep": "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
        "role_sep": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "begin_of_text_len": 1,
        "turn_seq_left_offset": -4,
    },
    "qwen": {
        "turn_sep": "<|im_start|>user\n",
        "role_sep": "<|im_end|>\n<|im_start|>assistant\n",
        "begin_of_text_len": 0,
        "turn_seq_left_offset": -4,
    },
}


@dataclasses.dataclass
class ChatTemplateOutput:
    raw: str = None
    encoded: BatchEncoding = None


def apply_chat_template(
    template,
    messages,
    system_message=None,
    tokenizer: PreTrainedTokenizer = None,
    add_generation_prompt=False,
    return_labels=False,
    **tokenization_kwargs,
):
    """
    Wrap the message using the template from fastchat according to its role

    Args:
        template: fastchat template name
        messages: a list of dictionaries, each of which is {'role': 'user/assistant', 'content': 'xxx'}
        system_message: system input
    """
    if len(tokenization_kwargs):
        assert (
            tokenizer is not None
        ), "Make sure the tokenizer is not None when passing tokenizer kwargs!"

    if template == "no":
        assert (
            tokenizer is not None
        ), "Make sure the tokenizer is not None when template is no!"

        prev_role = None
        conversation = ""

        for i, message in enumerate(messages):
            role = message["role"]
            content = message["content"]
            if prev_role == role:
                raise ValueError(
                    f"Current role (idx={i}) {role} and previous role {messages[i-1]['role']} are the same!"
                )

            if i == 0:
                content = tokenizer.decode(
                    tokenizer.encode(content), skip_special_tokens=True
                )
                user_message = content
            elif i == 1:
                # we use a space to separate user message and assistant response
                content = " " + content + tokenizer.eos_token
                assistant_message = content
            else:
                raise ValueError(
                    "Please use chat template when there are multi-turn conversations"
                )

            conversation += content

        encoded = tokenizer(conversation, **tokenization_kwargs)

        if return_labels:
            labels = encoded["input_ids"].copy()
            assistant_message_len = len(
                tokenizer.encode(assistant_message.lstrip(), add_special_tokens=False)
            )
            labels[:-assistant_message_len] = [
                -100 for _ in labels[:-assistant_message_len]
            ]
            encoded["labels"] = labels

            # sanity check
            for id_, label_ in zip(encoded["input_ids"], encoded["labels"]):
                assert (
                    id_ == label_ or label_ == -100
                ), "Found mismatch input_ids and labels!"

        return ChatTemplateOutput(raw=conversation, encoded=encoded)

    elif template == "hf":
        assert (
            return_labels is False
        ), "Returning labels with hf template is currently unsupported."
        tokenization_kwargs["return_dict"] = True
        raw = tokenizer.apply_chat_template(
            messages, add_generation_prompt=add_generation_prompt, tokenize=False
        )
        encoded = tokenizer.apply_chat_template(
            messages, add_generation_prompt=add_generation_prompt, **tokenization_kwargs
        )
        # for some tokenizer, the encoded input_ids are wrapped in a big list, while others are not
        if isinstance(encoded["input_ids"][0], list):
            for k, v in encoded.items():
                encoded[k] = v[0]
        return ChatTemplateOutput(raw=raw, encoded=encoded)

    conversation_template = get_conv_template(template)
    if system_message is not None:
        conversation_template.set_system_message(system_message)

    config = CHAT_TEMPLATE_CONFIG[template]

    role_map = {
        "user": conversation_template.roles[0],
        "assistant": conversation_template.roles[1],
    }
    prev_role = None

    for i, message in enumerate(messages):
        role = role_map[message["role"]]
        content = message["content"]
        if prev_role == role:
            raise ValueError(
                f"Current role (idx={i}) {role} and previous role {messages[i-1]['role']} are the same!"
            )
        conversation_template.append_message(role, content)
        prev_role = role

    if add_generation_prompt:
        assert (
            prev_role == role_map["user"]
        ), "You cannot add generation prompt after assistant output!"
        conversation_template.append_message(role_map["assistant"], None)

    conversation = conversation_template.get_prompt()

    if tokenizer is not None:
        encoded = tokenizer(conversation, **tokenization_kwargs)

        if return_labels:
            # Mask targets. Only compute loss on the assistant outputs.

            turn_sep = config["turn_sep"]
            role_sep = config["role_sep"]
            begin_of_text_len = config["begin_of_text_len"]
            turn_seq_left_offset = config["turn_seq_left_offset"]
            turn_sep_len = len(tokenizer.encode(turn_sep, add_special_tokens=False))

            # transform to array for fast value assignment
            labels = deepcopy(encoded["input_ids"])
            labels = np.array(labels)
            total_len = len(labels)

            turns = conversation.split(turn_sep)

            cur_len = 0
            for i, turn in enumerate(turns):
                if turn == "":
                    break

                turn_len = len(tokenizer(turn, add_special_tokens=False).input_ids)

                parts = turn.split(role_sep)

                if len(parts) == 2:
                    user_message, assistant_message = parts

                    user_message += role_sep
                    instruction_len = len(
                        tokenizer(user_message, add_special_tokens=False).input_ids
                    )

                    # for bos tokens
                    if i == 0:
                        turn_len += begin_of_text_len
                        instruction_len += begin_of_text_len

                    # Ignore the user instructions
                    labels[
                        max(cur_len + turn_seq_left_offset, 0) : cur_len
                        + instruction_len
                    ] = -100

                else:
                    labels[
                        max(cur_len + turn_seq_left_offset, 0) : cur_len
                        + turn_len
                        + turn_sep_len
                    ] = -100

                cur_len = cur_len + turn_len + turn_sep_len

                if cur_len > total_len:
                    break

            labels[max(cur_len + turn_seq_left_offset, 0) :] = -100

            encoded["labels"] = labels.tolist()

            # sanity check
            for id_, label_ in zip(encoded["input_ids"], encoded["labels"]):
                assert (
                    id_ == label_ or label_ == -100
                ), "Found mismatch input_ids and labels!"

    else:
        encoded = None

    return ChatTemplateOutput(raw=conversation, encoded=encoded)
