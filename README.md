# Data Generation using GPT-2 
Minimalist code to finetune a GPT-2 model to generate data for any use-case! 

Finetune the GPT-2 model on a small dataset. Use the finetuned model to generate similar data. 

## Finetuning the model
```
python -m train.py 

Following arguments can be passed to the script: 
 
--dataset: Provide the dataset
--epochs: Specify training epochs
--sample_every: produce sample output every n steps
--output_dir: Directory to store the finetuned model
```

## Generating data with the finetuned model 
```
python -m generator.py 

--model_dir: Specify the directory where the model is saved
--prompt: Add contextual data for the generator
```