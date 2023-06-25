import datetime
import os
import random
import time
import argparse

import numpy as np
import pandas
import torch
from torch.utils.data import DataLoader, random_split, RandomSampler, SequentialSampler

from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup

from dataloader import GPT2Dataset

parser = argparse.ArgumentParser()


def finetune(args):
    # Load the GPT tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>',
                                              pad_token='<|pad|>')  # gpt2-medium

    data = pandas.read_csv(args.dataset, header=None, encoding="utf-8", on_bad_lines='skip')
    data = data[0].copy()
    batch_size = 2

    # Prepare dataset for GPT-2 model
    dataset = GPT2Dataset(data, tokenizer, max_length=768)

    # Split into training and validation sets
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))

    # Create DataLoaders for training and validation datasets
    # Take training samples in random order.
    train_dataloader = DataLoader(train_dataset,
                                  sampler=RandomSampler(train_dataset),  # random batch sampling
                                  batch_size=batch_size)

    validation_dataloader = DataLoader(val_dataset,
                                       sampler=SequentialSampler(val_dataset),  # equential batch sampling
                                       batch_size=batch_size)

    configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)

    # instantiate the model
    model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)

    # necessary because of new tokens (e.g., bos_token) added to embeddings
    # otherwise the tokenizer and model tensors won't match
    model.resize_token_embeddings(len(tokenizer))

    device = torch.device("cuda")
    model.cuda()

    # Set seed value for reproducibility
    seed_val = 99

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    epochs = args.epochs
    learning_rate = 5e-4
    warmup_steps = 1e2
    epsilon = 1e-8

    # produce sample output every n steps
    sample_every = args.sample_every

    optimizer = AdamW(model.parameters(),
                      lr=learning_rate,
                      eps=epsilon)

    # Total training steps = [number of batches] x [number of epochs]
    # (Note that this is not the same as the number of training samples)
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler
    # This changes the learning rate as the training loop progresses
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    def format_time(elapsed):
        return str(datetime.timedelta(seconds=int(round((elapsed)))))

    total_t0 = time.time()
    training_stats = []
    model = model.to(device)

    for epoch_i in range(0, epochs):
        # ========================================
        #               Training
        # ========================================

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()
        total_train_loss = 0
        model.train()

        for step, batch in enumerate(train_dataloader):

            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)

            model.zero_grad()

            outputs = model(b_input_ids,
                            labels=b_labels,
                            attention_mask=b_masks,
                            token_type_ids=None)

            loss = outputs[0]

            batch_loss = loss.item()
            total_train_loss += batch_loss

            # Get sample every x batches.
            if step % sample_every == 0 and not step == 0:

                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataloader),
                                                                                         batch_loss, elapsed))

                model.eval()

                sample_outputs = model.generate(
                    bos_token_id=random.randint(1, 30000),
                    do_sample=True,
                    top_k=50,
                    max_length=200,
                    top_p=0.95,
                    num_return_sequences=1)

                for i, sample_output in enumerate(sample_outputs):
                    print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

                model.train()

            loss.backward()

            optimizer.step()

            scheduler.step()

        # Calculate the average loss over all batches
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure epoch time
        training_time = format_time(time.time() - t0)

        print("")
        print("Average training loss: {0:.2f}".format(avg_train_loss))
        print("Training epoch took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================

        print("")
        print("Running Validation...")

        t0 = time.time()

        model.eval()

        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)

            with torch.no_grad():
                outputs = model(b_input_ids,
                                # token_type_ids=None,
                                attention_mask=b_masks,
                                labels=b_labels)

                loss = outputs[0]

            batch_loss = loss.item()
            total_eval_loss += batch_loss

        avg_val_loss = total_eval_loss / len(validation_dataloader)

        validation_time = format_time(time.time() - t0)

        print("Validation Loss: {0:.2f}".format(avg_val_loss))
        print("Validation took: {:}".format(validation_time))

        # Record epoch stats
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

    output_dir = args.output_dir

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # torch.save(args, os.path.join(output_dir, 'training_args.bin'))


parser.add_argument("--dataset", type=str, default="data.txt", help="Provide the dataset")
parser.add_argument("--epochs", type=int, default="5", help="Specify training epochs")
parser.add_argument("--sample_every", type=int, default="5", help="produce sample output every n steps")
parser.add_argument("--output_dir", type=str, default="models/", help="Directory to store the finetuned model")

args = parser.parse_args()

finetune(args)
