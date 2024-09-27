# ---- Library Imports ----
# Importing PyTorch, a deep learning framework, which is used for tensor operations and running models on GPUs.
import torch

# Importing the GPT2 model and tokenizer from Hugging Face's transformers library.
# GPT2LMHeadModel is the specific GPT-2 model that is designed for language modeling tasks (i.e., predicting the next word in a sequence).
# GPT2Tokenizer is the tokenizer that converts text into tokens that the GPT-2 model understands.
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

# Importing Dataset from Hugging Face's datasets library.
# This library allows easy handling of large datasets and integrates with Hugging Face's transformers for model training.
from datasets import Dataset

# Importing pandas for handling and manipulating data in DataFrame format (like reading CSV files).
import pandas as pd

# ---- Step 1: Load and Prepare Data ----
def load_and_prepare_data(file_path):
    # Load your dataset from a CSV file using pandas.
    # This assumes the CSV contains a column 'Cleaned_Abstract' which holds pre-processed text data.
    df = pd.read_csv(file_path)
    
    # Convert the pandas DataFrame into a Hugging Face Dataset object.
    # Hugging Face models work more efficiently with Dataset objects.
    dataset = Dataset.from_pandas(df[['Cleaned_Abstract']])

    # Load the GPT-2 tokenizer from Hugging Face's pre-trained models.
    # The tokenizer converts raw text into tokens that the model can work with.
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Set the padding token to be the same as the end-of-sequence (EOS) token.
    # This is important for handling sequences shorter than the maximum length during batch processing.
    tokenizer.pad_token = tokenizer.eos_token

    # Define a function to tokenize the text data.
    def tokenize_function(examples):
        # Tokenize the "Cleaned_Abstract" text, padding or truncating it to a max length of 512 tokens.
        tokens = tokenizer(examples["Cleaned_Abstract"], padding="max_length", truncation=True, max_length=512)
        
        # For language modeling, the labels are the same as the input tokens (the model predicts the next token).
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    # Apply the tokenization function to the entire dataset in batches.
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    return tokenized_dataset, tokenizer

# ---- Step 2: Fine-Tune GPT-2 Model ----
def fine_tune_gpt2(dataset, tokenizer):
    # Check if a GPU is available. If so, set the device to GPU, else use CPU.
    # Using a GPU accelerates training significantly, especially with large models like GPT-2.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load the pre-trained GPT-2 model for language modeling.
    # GPT2LMHeadModel is specifically designed for tasks where the model predicts the next token in a sequence.
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # Move the model to the appropriate device (GPU or CPU).
    model.to(device)

    # Set up the training arguments to control the training process.
    training_args = TrainingArguments(
        output_dir="./results",  # Directory where model checkpoints and results will be saved.
        overwrite_output_dir=True,  # Overwrite the output directory if it already exists.
        num_train_epochs=3,  # Train the model for 3 epochs (i.e., three full passes over the dataset).
        per_device_train_batch_size=4,  # Batch size per GPU during training (adjust based on available memory).
        per_device_eval_batch_size=4,  # Batch size for evaluation (here we use the same batch size as training).
        evaluation_strategy="epoch",  # Evaluate the model at the end of each epoch.
        save_steps=1000,  # Save a checkpoint of the model every 1000 steps during training.
        logging_dir='./logs',  # Directory where the training logs will be stored.
        logging_steps=10,  # Log progress every 10 steps.
    )

    # Initialize the Trainer class, which handles the training loop and evaluation.
    # The Trainer abstracts away much of the complexity involved in the training loop.
    trainer = Trainer(
        model=model,  # The pre-trained GPT-2 model that we are fine-tuning.
        args=training_args,  # Training arguments we defined earlier.
        train_dataset=dataset,  # The tokenized dataset used for training.
        eval_dataset=dataset,  # The same dataset used for evaluation (usually a separate validation set is used, but for simplicity, we use the same here).
    )

    # Begin the fine-tuning process. This will adjust the pre-trained model's weights to better fit the new dataset.
    trainer.train()

    # Save the fine-tuned model and tokenizer so that they can be reused for inference or further training later.
    model.save_pretrained("./fine-tuned-gpt2")
    tokenizer.save_pretrained("./fine-tuned-gpt2")

    print("Fine-tuning complete! Model saved to './fine-tuned-gpt2'")

# ---- Main Execution ----
if __name__ == "__main__":
    # Specify the file path of the dataset containing medical publication abstracts.
    # The dataset should be preprocessed and saved as a CSV file with a 'Cleaned_Abstract' column.
    file_path = "cleaned_pubmed_abstracts.csv"

    # Step 1: Load and tokenize the dataset to prepare it for model training.
    tokenized_dataset, tokenizer = load_and_prepare_data(file_path)

    # Step 2: Fine-tune the GPT-2 model on the tokenized dataset.
    fine_tune_gpt2(tokenized_dataset, tokenizer)
