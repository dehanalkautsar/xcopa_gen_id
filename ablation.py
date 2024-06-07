# Modified from: https://github.com/LazarusNLP/IndoT5/blob/main/scripts/run_qa.py

from dataclasses import dataclass
from datargs import parse

import math
import time
import evaluate
import numpy as np
from datasets import load_dataset
from transformers.trainer_utils import EvalPrediction
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)

from utils import QuestionAnsweringSeq2SeqTrainer

##################################
# CONFIGURATIONS
# ################################
@dataclass
class Args:
    model_checkpoint: str = "LazarusNLP/IndoNanoT5-base"
    dataset_name: str = "dehanalkautsar/xcopa_gen_id"
    answer_column_name: str = "label"
    input_max_length: int = 128
    target_max_length: int = 128
    num_beams: int = 5
    output_dir: str = "outputs/ablation-1percentdata-indot5-lr5e5-xcopagenid"
    num_train_epochs: int = 10
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01
    optim: str = "adamw_torch_fused"
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    dataset_size: float = 1 
    dataset_noise: float = 0 
    freeze_attention: bool = True
""
def main(args: Args):
    # load dataset, tokenizer, model
    TRAIN_LEN = 400 # See dehanalkautsar/xcopa_gen_id
    len_data_train = int(args.dataset_size*TRAIN_LEN)
    len_data_noise = int(args.dataset_noise*TRAIN_LEN)
    print(f"LEN DATA TRAIN: {len_data_train}")
    print(f"LEN DATA NOISE: {len_data_noise}")
    if len_data_noise:
        train_dataset, validation_dataset, test_dataset = load_dataset(args.dataset_name, split=[f"train[:{len_data_train}]+train[-{len_data_noise}:]", "validation[:50%]", "test[:50%]"], trust_remote_code=True)
    else:
        train_dataset, validation_dataset, test_dataset = load_dataset(args.dataset_name, split=[f"train[:{len_data_train}]", "validation[:50%]", "test[:50%]"], trust_remote_code=True)
        
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint)
    if args.freeze_attention: #freezing attention (self & cross) blocks
        # All selfattention in the encoder
        modules_to_freeze = [model.encoder.block[i].layer[0] for i in range(len(model.encoder.block))]
        # And the decoder modules, which has both a SelfAttention (layer[0]) 
        modules_to_freeze.extend([model.decoder.block[i].layer[0] for i in range(len(model.decoder.block))])
        # and CrossAttention (layer[1]) block
        modules_to_freeze.extend([model.decoder.block[i].layer[1] for i in range(len(model.decoder.block))])
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False #freeze!
        
    def preprocess_function(examples):
        inputs, targets = examples['input'], examples['label']

        model_inputs = tokenizer(inputs, max_length=args.input_max_length, truncation=True)
        labels = tokenizer(text_target=targets, max_length=args.target_max_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def preprocess_validation_function(examples):
        inputs, targets = examples['input'], examples['label']

        model_inputs = tokenizer(
            inputs,
            max_length=args.input_max_length,
            truncation=True,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )
        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=args.target_max_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    tokenized_train_dataset = train_dataset.map(
        preprocess_function, batched=True, remove_columns=train_dataset.column_names
    )
    tokenized_validation_dataset = validation_dataset.map(
        preprocess_validation_function, batched=True, remove_columns=validation_dataset.column_names
    )
    tokenized_test_dataset = test_dataset.map(
        preprocess_validation_function, batched=True, remove_columns=test_dataset.column_names
    )

    # prepare s2s collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=args.model_checkpoint, label_pad_token_id=tokenizer.pad_token_id
    )

    # SQuAD v2 and BLEU metric for evaluation
    squad_v2 = evaluate.load("squad_v2")
    bleu = evaluate.load("bleu")

    def compute_metrics(p: EvalPrediction):
        squad_res = squad_v2.compute(predictions=p.predictions, references=p.label_ids)
        preds = []
        refs = []
        for prediction, reference in zip(p.predictions, p.label_ids):
          preds.append(prediction["prediction_text"])
          refs.append(reference["answers"]["text"])
        bleu_res = bleu.compute(predictions=preds, references=refs)
        return {"exact":squad_res['exact'], "f1":squad_res['f1'], 'bleu':bleu_res['bleu'], 'bleu1':bleu_res['precisions'][0], 'bleu2':bleu_res['precisions'][1], 'bleu3':bleu_res['precisions'][2], 'bleu4':bleu_res['precisions'][3]}

    def post_processing_function(examples, features, outputs, stage="eval"):
        # Decode the predicted tokens.
        preds = outputs.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        formatted_predictions = []
        references = []
        for i, ex in enumerate(examples):
          if decoded_preds[i] == '':
            decoded_preds[i] = 'nothing'
          formatted_predictions.append({"id": str(i), "prediction_text": decoded_preds[i], "no_answer_probability":0.0})
          references.append({"id": str(i), "answers": {"answer_start": [0], "text": [ex[args.answer_column_name].lower()]}})
        print(formatted_predictions[:1])
        print(references[:1])
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    callbacks = [EarlyStoppingCallback(args.early_stopping_patience, args.early_stopping_threshold)]

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        optim=args.optim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        save_total_limit=2,
        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    trainer = QuestionAnsweringSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_validation_dataset,
        eval_examples=validation_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        post_process_function=post_processing_function,
    )

    trainer.train()

    result = trainer.evaluate(tokenized_test_dataset, test_dataset, max_length=args.target_max_length, num_beams=args.num_beams)
    print(result)

if __name__ == "__main__":
    args = parse(Args)
    main(args)
