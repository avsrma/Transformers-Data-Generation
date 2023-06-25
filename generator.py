import argparse

import torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# #######################
# # GENERATION
# #######################

parser = argparse.ArgumentParser()


def sentence_generation(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_dir = args.model_dir

    # Load a trained model and vocabulary that you have fine-tuned
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model.to(device)

    model.eval()

    prompt = args.prompt

    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    generated = generated.to(device)

    print(generated)

    sample_outputs = model.generate(generated,
                                    # bos_token_id=random.randint(1,30000),
                                    do_sample=True,
                                    top_k=50,
                                    max_length=300,
                                    top_p=0.95,
                                    num_return_sequences=30
                                    )

    for i, sample_output in enumerate(sample_outputs):
        print("{}: {}\n\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))


parser.add_argument("--model_dir", type=str, default="models/", help="Specify the directory where the model is saved")
parser.add_argument("--prompt", type=str, default="<|startoftext|>", help="Add contextual data for the generator")

args = parser.parse_args()

sentence_generation(args)
