# -*- coding: utf-8 -*-
import json
from dataclasses import dataclass, field
from typing import Dict, Optional
import jsonlines
import torch
import transformers
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="openbmb/MiniCPM-2B-sft-bf16")


@dataclass
class DataArguments:
    train_data_path: str = field(
        default="data/AdvertiseGenChatML/train.json",
        metadata={"help": "Path to the training data."},
    )
    eval_data_path: str = field(
        default="data/AdvertiseGenChatML/dev.json",
        metadata={"help": "Path to the test data."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = field(default=False)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    
    def __init__(
        self,
        data_path,
        tokenizer,
        model_max_length=4096,
        user_tokens='<用户>',
        assistant_tokens='<AI>',
    ):
        super(SupervisedDataset, self).__init__()
        # self.data = json.load(open(data_path))
        self.data = []
        with jsonlines.open(data_path, "r") as fin:
            for line in fin:
                self.data.append(line)
        fin.close()
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.user_tokens = self.tokenizer.encode(user_tokens) # The id of <USER> which is suitable for different models.
        self.assistant_tokens = self.tokenizer.encode(assistant_tokens) # The id of <AI> which is suitable for different models.
        self.ignore_index = -100
        item = self.preprocessing(self.data[0])
        print("input:", self.tokenizer.decode(item["input_ids"]))
        labels = []
        for id_ in item["label_ids"]:
            if id_ == -100:
                continue
            labels.append(id_)
        print("label:", self.tokenizer.decode(labels))

    def __len__(self):
        return len(self.data)

    # This function is used to preprocess the data.
    # Your task is to build the input_ids and label_ids from the example.
    # The input_ids should be the input of the model, and the label_ids should be the target output of the model.

    # HINT 1: If you are unsure about label_ids, remember that our training task is NEXT TOKEN PREDICTION!!!
    #         Each label_id should correspond to the next token that we want the model to predict.

    # HINT 2: The <USER> and <AI> tokens should be added to the input_ids. However, their corresponding label_ids should be an ignorable id.

    # HINT 3: For this task, the label_ids should be ignorable for the content following <NAME>.
    #         <NAME> can be either <USER> or <AI>. Decide which content should be ignored accordingly.

    # After building the input_ids and label_ids, you also need to build the attention_mask.

    # HINT 1: The attention should be 1 for each token in a sentence.
    # HINT 2: Handle padding carefully to ensure that the attention mask correctly reflects the actual content length.
    # HINT 3: The attention_mask is initially a sequence of 1s and 0s. 
    #         Later, another part of the code will convert this sequence into a matrix form.

    # Example of attention_mask transformation:
    # [1, 0, 1, 1] -> [ [1, 0, 0, 0],
    #                   [0, 0, 0, 0],
    #                   [0, 0, 1, 1],
    #                   [0, 0, 1, 1] ]

    def preprocessing(self, example):
        input_ids = [self.tokenizer.bos_token_id]
        label_ids = [self.ignore_index]

        ##### build input_ids and label_ids from example
        ## add you code (10 - 20 lines)
        content = example["content"]
        summary = example["summary"]

        input_ids.extend(self.user_tokens)
        input_ids.extend(tokenizer.encode(content))
        input_ids.extend(self.assistant_tokens)
        input_ids.extend(tokenizer.encode(summary))
        # input_ids.append(self.tokenizer.eos_token_id)


        # label_ids.append(self.ignore_index)
        # [label_ids.append(r) for r in tokenizer(content)[1:]]
        # # The last to predict label should be ignored???
        # label_ids.append(self.ignore_index)
        # label_ids.append(self.assistant_tokens)

        ignore = [self.ignore_index for i in range(len(self.user_tokens) + len(tokenizer.encode(content)))]
        label_ids.extend(ignore)
        label_ids.extend(self.assistant_tokens)
        label_ids.extend(tokenizer.encode(summary))

        ##### build input_ids and label_ids from example

        input_ids.append(self.tokenizer.eos_token_id)
        label_ids.append(self.tokenizer.eos_token_id)
        # truncate to max len
        input_ids = input_ids[: self.model_max_length]
        label_ids = label_ids[: self.model_max_length]
        ##### build attention mask
        ## add you code (less than 3 lines)
        attention_mask = [1 for i in range(len(input_ids))]
        ##### build attention mask
        # pad to max len
        len_in = len(input_ids)
        input_ids += [self.tokenizer.eos_token_id] * (
            self.model_max_length - len(input_ids)
        )

        label_ids += [self.ignore_index] * (self.model_max_length - len(label_ids))
        ##### update attention mask for padding
        ## add you code (less than 3 lines)
        to_add = [0 for i in range(self.model_max_length - len_in)]
        attention_mask.extend(to_add)
        ##### update attention mask for padding
        # convert to pt tensor
        input_ids = torch.LongTensor(input_ids)
        label_ids = torch.LongTensor(label_ids)
        attention_mask = torch.LongTensor(attention_mask)
        return {
            "input_ids": input_ids,
            "label_ids": label_ids,
            "attention_mask": attention_mask,
        }

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.preprocessing(self.data[idx])

# In this function, you need to load the model and tokenizer.

# HINT: Refer to the beginning of this code to see which libraries have already been imported.
#       You do not need to import any additional libraries.
#       This means you should use one of the libraries we have already provided.

def load_model_and_tokenizer(
    model_path: str,
    max_length: int = 4096,
    use_lora: bool = True,
    bf16: bool = False,
    fp16: bool = False,
):
    """load model and tokenizer"""
    ##### load tokenizer
    ## add you code (less than 3 lines)
    checkpoint = "openbmb/MiniCPM-1B-sft-bf16"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    ##### load tokenizer finished
    tokenizer.pad_token = tokenizer.eos_token

    assert not (bf16 and fp16), "bf16 or fp16, not both"
    if bf16:
        dtype = torch.bfloat16
    elif fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    ##### load model
    ## add you code (5 - 10 lines)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True, torch_dtype=torch.bfloat16)
    ##### load model finished
    if use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        lora_config = LoraConfig(
            init_lora_weights="gaussian",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj"],
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        # trainable params: 2,949,120 || all params: 3,010,652,928 || trainable%: 0.09795616002669305
        model.print_trainable_parameters()
        # model.enable_input_require_grads()  # need when using adapter

    return model, tokenizer


if __name__ == "__main__":
    model_path = "/mnt/data/user/tc_agi/yh/models/MiniCPM"
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model, tokenizer = load_model_and_tokenizer(
        model_path=model_args.model_name_or_path,
        max_length=training_args.model_max_length,
        use_lora=training_args.use_lora,
        bf16=training_args.bf16,
        fp16=training_args.fp16
    )

    train_dataset = SupervisedDataset(
        data_path=data_args.train_data_path,
        tokenizer=tokenizer,
        model_max_length=training_args.model_max_length,
    )
    eval_dataset = SupervisedDataset(
        data_path=data_args.eval_data_path,
        tokenizer=tokenizer,
        model_max_length=training_args.model_max_length,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    # save the incremental PEFT weights, more details can be found in https://huggingface.co/blog/peft
    # model.save_pretrained("output_dir")