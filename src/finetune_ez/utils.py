from typing import List, Dict
import logging
import os
import math
import torch.backends
import torch.backends.cudnn
import tqdm

import numpy as np
import evaluate
import torch
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification
from sklearn.utils.class_weight import compute_class_weight
from transformers.modeling_utils import unwrap_model
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)

import torch
import random
import numpy as np

def set_seed(seed):
    random.seed(seed) # Pyhon random module seed
    np.random.seed(seed) # Numpy seed for generating random numbers
    torch.manual_seed(seed) # PyTorch seed for CPU
    torch.cuda.manual_seed(seed) # PyTorch seed for CUDA
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior (note: can slow down performance)
    torch.backends.cudnn.benchmark = False  # Disable the benchmark for deterministic results

def get_compute_classifier_metrics(num_labels):
    """
    Returns a function to compute classification metrics (accuracy and F1-score).

    Args:
        num_labels (int): The number of labels in the classification task. Determines the type of averaging.

    Returns:
        function: A function to compute the metrics given logits and labels.
    """
    # Log the start of the metric loading process
    logger.info("Metrics been loaded")

    # Determine the type of averaging to use for metrics:
    # - 'binary' for binary classification tasks (num_labels == 2).
    # - 'macro' for multi-class classification tasks (num_labels > 2).
    if num_labels == 2:
        average = 'binary'
    else:
        average = 'macro'

    # Load the accuracy metric with the specified averaging method
    load_accuracy = evaluate.load('accuracy', average=average)
    # Load the F1 metric with the specified averaging method
    load_f1 = evaluate.load('f1', average=average)

    # Define the function to compute metrics using predictions and labels
    def this_compute_classifier_metrics(eval_pred):
        """
        Computes accuracy and F1-score for the given logits and labels.

        Args:
            eval_pred (tuple): A tuple containing:
                - logits (np.ndarray): The raw output predictions from the model.
                - labels (np.ndarray): The true labels for the dataset.

        Returns:
            dict: A dictionary containing the computed accuracy and F1-score.
        """
        logits, labels = eval_pred  # Unpack logits and labels
        # Get predicted labels by selecting the class with the highest logit value
        predictions = np.argmax(logits, axis=-1)
        # Compute accuracy metric
        accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
        # Compute F1-score metric
        f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
        # Return the metrics as a dictionary
        macro_f1 = f1_score(labels, predictions, average='macro')
        return {"accuracy": accuracy, "f1": f1, "macro_f1": macro_f1}

    # Log that the metrics function has been successfully loaded
    logger.info("Metrics are loaded")

    # Return the function that computes the metrics
    return this_compute_classifier_metrics


def compute_model_metrics(model, dataset, data_collator, batch_size, prefix="", compute_classifier_metrics_func=None):
    """
    Computes evaluation metrics (e.g., accuracy, F1-score, loss) for a model on a given dataset.

    Args:
        model: The PyTorch model to be evaluated.
        dataset: The dataset to evaluate the model on (e.g., validation or test dataset).
        data_collator: A function to collate and prepare batches of data.
        batch_size (int): The number of samples per batch during evaluation.
        prefix (str): A string to prefix metric names (e.g., "val_" or "test_").
        compute_classifier_metrics_func (function, optional): A function to compute classification metrics 
            (e.g., accuracy, F1-score). Defaults to a function based on the number of labels.

    Returns:
        dict: A dictionary containing the computed metrics with prefixed keys.
    """
    # Initialize containers for storing losses, logits (predictions), and labels
    all_losses = []
    all_logits, all_labels = [], []

    # Create a DataLoader to process the dataset in batches
    for batch in tqdm.tqdm(torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)):
        with torch.no_grad():  # Disable gradient computation for inference to save memory and speed up processing
            # Handle multi-GPU models (DataParallel) vs single-GPU models
            if type(model) is torch.nn.DataParallel:
                out = model(**{k: v.to(f'cuda:{model.device_ids[0]}') for k, v in batch.items()})
            else:
                out = model(**{k: v.to(model.device) for k, v in batch.items()})

        # Extract labels and remove them from the batch
        labels = batch.pop("labels")
        all_labels.append(labels)  # Collect true labels
        all_logits.append(out.logits)  # Collect logits (model predictions)
        # Store loss for each input in the batch, converted to CPU
        all_losses += [out.loss.cpu()] * len(batch['input_ids'])

    # If no custom metrics function is provided, use a default function based on the number of labels
    if compute_classifier_metrics_func is None:
        compute_classifier_metrics_func = get_compute_classifier_metrics(model.num_labels)

    # Compute metrics using the logits and labels
    metrics = compute_classifier_metrics_func(
        (torch.vstack(all_logits).cpu(), torch.hstack(all_labels).cpu())
    )
    # Compute and add the average loss to the metrics
    metrics['loss'] = float(torch.mean(torch.hstack(all_losses)))

    # Add prefix to all metric keys and return the metrics dictionary
    return {f'{prefix}{k}': v for (k, v) in metrics.items()}


def log_round_metrics(model, val_dataset, data_collator, inference_batch, round: int, selected_exes: List[int],
                      test_data=None, compute_classifier_metrics_func=None) -> Dict:
    """
    Logs evaluation metrics for a specific training round, including validation and optional test metrics.

    Args:
        model: The PyTorch model to evaluate.
        val_dataset: The validation dataset.
        data_collator: A function to collate data into batches for the model.
        inference_batch (int): Batch size for inference.
        round (int): The current training round (e.g., epoch or iteration number).
        selected_exes (List[int]): A list of selected examples for the current round.
        test_data (optional): The test dataset to evaluate on (default: None).
        compute_classifier_metrics_func (optional): A function to compute metrics like accuracy or F1-score.

    Returns:
        Dict: A dictionary containing computed metrics for the validation and (optionally) test datasets,
              along with the training round and selected examples.
    """
    # Set the model to evaluation mode
    model.eval()

    # Move the model to GPU if available
    if torch.cuda.is_available():
        if type(model) is not torch.nn.DataParallel and torch.cuda.device_count() > 1:
            # Use DataParallel if multiple GPUs are available
            model = torch.nn.DataParallel(model.to('cuda'))
        elif torch.cuda.device_count() == 1:
            # Move model to the single GPU
            model.to('cuda')

    # Log the start of validation metrics computation
    logger.info("Computing val metrics")
    # Compute validation metrics using the `compute_model_metrics` function
    metrics = compute_model_metrics(model, val_dataset, data_collator,
                                    inference_batch, prefix='val_', compute_classifier_metrics_func=compute_classifier_metrics_func)

    # Add the current training round to the metrics
    metrics['round'] = round

    # If test data is provided, compute and log test metrics
    if test_data is not None:
        logger.info("Computing test metrics")
        metrics.update(compute_model_metrics(model, test_data, data_collator,
                                             inference_batch, prefix='test_', compute_classifier_metrics_func=compute_classifier_metrics_func))

    # Log the selected examples for the round
    metrics['selected_ex'] = selected_exes

    # Move the model back to the CPU to free GPU memory
    model.to('cpu')

    # Return the computed metrics
    return metrics


def process_dataset(dataset, tokenizer):
    """
    Processes a dataset to ensure it is tokenized and compatible with a model's input requirements.
    
    Args:
        dataset: A dataset object (e.g., Hugging Face Dataset) to process.
        tokenizer: A tokenizer object used to tokenize the text data.
    
    Returns:
        A processed dataset with tokenized inputs ('input_ids', 'attention_mask') 
        and labels ('labels') ready for model training or evaluation.
    """
    # Check if the dataset has already been tokenized (contains 'input_ids')
    if 'input_ids' not in dataset.column_names:
        logger.info("Processing required, doing that")
        
        # Define a preprocessing function to tokenize the text data
        def preprocess_function(examples):
            # Tokenize text with truncation and padding to a multiple of 8 for GPU efficiency
            return tokenizer(examples["text"], 
                             truncation=True, 
                             max_length=min(tokenizer.max_len_single_sentence, 512), 
                             pad_to_multiple_of=8)

        # Apply the preprocessing function to the dataset in a batched manner
        tokenized_train = dataset.map(preprocess_function, batched=True)

        # If the dataset contains a "label" column, rename it to "labels" for consistency
        if "label" in tokenized_train.column_names:
            tokenized_train = tokenized_train.rename_columns({"label": "labels"})
        
        # Remove the original "text" column after tokenization (not needed anymore)
        return tokenized_train.remove_columns("text")
    
    else:
        # If the dataset is already tokenized, log that no additional processing is required
        logger.info("Already processed")
        
        # Rename the "label" column to "labels" if it exists, to ensure compatibility
        if "label" in dataset.column_names:
            dataset = dataset.rename_columns({"label": "labels"})

        # Return the dataset as is since it has already been processed
        return dataset


def train_model_with_hf(model, training_dataset, validation_dataset, data_collator, optimizer_cls, output_dir: str, batch_size: int=32, n_epochs=32,
                        lr: float = 2e-5, lr_plateau_patience: int = None, lr_plateau_factor: int = 0.1, compute_classifier_metrics_func=None,
                        problem_type = None, test_dataset=None, gradient_checkpointing=False, reweight=False):
    """
    Trains a PyTorch model using a Hugging Face Dataset.
    """

    # Initialize the optimizer with the provided optimizer class and learning rate
    optimizer = optimizer_cls(model.parameters(), lr=lr)

    # Set default early stopping patience
    early_stopping_patience = 3

    # If a learning rate plateau scheduler is provided, configure it
    if lr_plateau_patience:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                  factor=lr_plateau_factor, 
                                                                  patience=lr_plateau_patience)
        # Adjust early stopping patience to account for plateau patience
        early_stopping_patience = lr_plateau_patience * 2
    else:
        lr_scheduler = None

    # Combine optimizer and scheduler into a tuple
    optimizers = (optimizer, lr_scheduler)

    # Default metric computation function if none is provided
    if compute_classifier_metrics_func is None:
        compute_classifier_metrics_func = get_compute_classifier_metrics(model)
    
    # Determine the problem type if not explicitly provided
    if problem_type is None:
        problem_type = model.get_problem_type()

    # Define training arguments for the Hugging Face Trainer
    training_args = TrainingArguments(
        output_dir=output_dir,  # Directory to save model checkpoints and logs
        evaluation_strategy='epoch',  # Evaluate at the end of each epoch
        learning_rate=lr,  # Initial learning rate
        per_device_train_batch_size=batch_size,  # Training batch size per device
        per_device_eval_batch_size=batch_size,  # Evaluation batch size per device
        num_train_epochs=n_epochs,  # Number of training epochs
        warmup_steps=0,  # No warmup steps for learning rate
        bf16=False,  # Disable bfloat16 precision
        do_eval=True,  # Perform evaluation during training
        report_to=['tensorboard'],  # Log metrics to TensorBoard
        save_total_limit=1,  # Keep only the latest checkpoint
        save_strategy="epoch",  # Save checkpoints at the end of each epoch
        metric_for_best_model='eval_macro_f1',  # Metric to determine the best model
        greater_is_better=True,  # Higher values of the metric are better
        load_best_model_at_end=True,  # Load the best model after training
        remove_unused_columns=False,  # Do not remove unused dataset columns
        gradient_checkpointing=gradient_checkpointing  # Enable gradient checkpointing if specified
    )

    # Handle class reweighting for imbalanced datasets
    if not reweight:
        rw = None  # No class reweighting if not specified
    else:
        # Compute class weights using the 'balanced' strategy
        class_weights = compute_class_weight(class_weights='balanced',
                                              classes=np.unique(training_dataset['labels']),
                                              y=training_dataset['labels'])
        # Optional: Adjust specific class weights manually (commented out here)
        # adjustment_factor_2 = 1.5
        # adjustment_factor_3 = 5.0
        # class_weights[1] *= adjustment_factor_2
        # class_weights[2] *= adjustment_factor_3

        rw = class_weights

    # Wrap the Hugging Face Trainer with a custom decorator to modify loss computation
    Trainer_CLS = compute_loss_hf_trainer_decorator(Trainer, model.num_labels, problem_type, rw)

    # Instantiate the customized Trainer
    trainer = Trainer_CLS(
        model=model,  # The model to be trained
        args=training_args,  # Training arguments defined earlier
        optimizers=optimizers,  # Tuple of optimizer and learning rate scheduler
        train_dataset=training_dataset,  # Training dataset
        eval_dataset=validation_dataset,  # Validation dataset for evaluation during training
        data_collator=data_collator,  # Function to preprocess batches of data
        compute_metrics=compute_classifier_metrics_func,  # Function to compute evaluation metrics
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]  # Early stopping callback
    )

    # Start the training process
    trainer.train()

    if test_dataset:
        return trainer.model, trainer.evaluate(test_dataset, metric_key_prefix='test')
    else:
        return trainer.model
    

def run_train_model_with_hf(model_str: str, training_dataset, validation_dataset, optimizer_cls,
                            output_dir: str, batch_size: int=32, n_epochs=32, lr: float = 2e-5,
                            lr_plateau_patience: int=None, lr_plateau_factor: int=0.1, tokenizer=None,
                            save_initialization=False, num_labels: int=2, problem_type='single_label_classification', reweight=False):
    """
    Runs the training process using a Hugging Face Dataset and a provided model string.
    """
    # Load the tokenizer if not provided
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_str)

    # Create a data collator that ensures padding to a multiple of 64 for efficient GPU usage
    data_collator = DataCollatorWithPadding(tokenizer, return_tensors='pt', pad_to_multiple_of=64)

    # Process training and validation datasets with the tokenizer
    # This prepares the datasets for input to the model (e.g., tokenization, encoding labels)
    training_dataset = process_dataset(training_dataset, tokenizer)
    validation_dataset = process_dataset(validation_dataset, tokenizer)

    # Initialize the pre-trained model with the specified number of labels and problem type
    # Currently designed for single-label classification
    model = make_reload_func(model_str, num_labels, problem_type)()

    # Optionally save the initialized model (useful for debugging or reproducing experiments)
    if save_initialization:
        model.save_pretrained(os.path.join(output_dir, 'initialized-model'))

    # Call the `train_model_with_hf` function to handle the training process
    # Pass all necessary parameters including datasets, model, and optimizer configuration
    
    return train_model_with_hf(model, training_dataset, validation_dataset, data_collator, optimizer_cls, os.path.join(output_dir, 'training'),
                               batch_size=batch_size, n_epochs=n_epochs, lr=lr, lr_plateau_patience=lr_plateau_patience, lr_plateau_factor=lr_plateau_factor,
                               problem_type=problem_type, reweight=reweight
                               )


def compute_loss_hf_trainer_decorator(trainer, num_labels, problem_type, rw=None):
    """
    A decorator function that conditionally returns either the base `trainer` or a custom subclass
    that overrides `compute_loss`. The subclass can apply a reweighted loss if `rw` is provided.
    
    Args:
        trainer (Trainer): The original Hugging Face Trainer class to subclass or return unmodified.
        num_labels (int): The number of labels/classes (currently unused in this code).
        problem_type (str): A string indicating the type of problem (e.g., 'soft_single_label_classification').
        rw (list[float] | None): Class weight multipliers for reweighting. If None, no reweighting is applied.
    
    Returns:
        Trainer | SubClass: Either the original Trainer class or a subclass that overrides `compute_loss`.
    """

    # If the problem is NOT 'soft_single_label_classification' AND rw is None,
    # simply return the original trainer without any custom logic.
    # This means no custom subclass is created if we don't need it.
    if problem_type not in ['soft_single_label_classification', 'max_exponential_margin'] and rw is None:
        return trainer

    class SubClass(trainer):
        """
        A custom subclass of the original Trainer that overrides `compute_loss` to allow
        reweighted loss when `rw` is not None.
        """
        def compute_loss(self, model, inputs, return_outputs=False):
            """
            Override the default compute_loss method to optionally apply a reweighted CrossEntropyLoss.
            
            Args:
                model (nn.Module): The model to be trained.
                inputs (dict): Batch of data, expected to have 'labels' and other keys used by the model.
                return_outputs (bool): If True, return (loss, model_outputs). Otherwise, return just the loss.
            
            Returns:
                loss | (loss, output): Depending on `return_outputs`.
            """
            labels = None
            loss = None

            # Extract labels from inputs if present (and remove them so they don't go into the model twice).
            if 'labels' in inputs:
                labels = inputs.pop('labels')
            
            # Run a forward pass on the remaining inputs to get the outputs (including logits).
            output = model(**inputs)
            if problem_type == "soft_single_label_classification":
                loss = calculate_soft_single_label_classifiction(labels, output.logits, num_labels)

            elif problem_type == 'max_exponential_margin': # no bias!
                loss = calculate_exponential_loss(labels, output.logits)

            # If rw is provided, use that to create a reweighted CrossEntropyLoss.
            elif rw is not None:
                loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor(rw))
                loss = loss_fct(output.logits, labels)

            # Return (loss, output) if return_outputs is True, else return just the loss.
            return (loss, output) if return_outputs else loss

    # Return the custom subclass instead of the original trainer.
    return SubClass


'''
THis make reload function is being used for two purposes. IF the problem type is single label, then we just return the normal pretrained model else
when the problem type is not single label, thats when we need custom loss function and customer decorator for it.
'''

def make_reload_func(dir_: str, num_labels: int, problem_type, for_training=True):
    """
    Creates a reload function to dynamically load and configure a Hugging Face model for sequence classification.
    
    Parameters:
    dir_ (str): Path to the directory containing the pre-trained model.
    num_labels (int): Number of output labels/classes for the classification task.
    problem_type (str): Specifies the type of classification task (e.g., 'single_label_classification').
    for_training (bool): If True, the model is loaded for training. If False, the model is configured for evaluation.

    Returns:
    reload_func (function): A function that, when called, loads and returns the configured model.
    """
    def reload_func():
        """
        Dynamically reloads the model with the specified configurations.
        """

        # Check if the problem type requires alternative/custom loss handling
        if problem_type in ALTERNATIVE_LOSSES:
            # For alternative losses, temporarily use 'single_label_classification' to ensure compatibility
            fake_problem_type = 'single_label_classification'
            model = AutoModelForSequenceClassification.from_pretrained(
                dir_, num_labels=num_labels, problem_type=fake_problem_type
            )

            # If the model is not for training, customize the forward and __call__ methods
            if not for_training:
                # Wrap the forward method with a custom decorator for alternative loss handling
                model.forward = alternative_loss_decorator(model, model.forward, problem_type)
                # Similarly, wrap the __call__ method to ensure consistent behavior
                model.__call__ = alternative_loss_decorator(model, model.__call__, problem_type)

            # Special handling for the 'max_exponential_margin' loss
            if problem_type == 'max_exponential_margin':  # Ensure no bias term is used
                model.classifier.out_proj.bias = torch.nn.Parameter(
                    torch.zeros(model.classifier.out_proj.bias.shape), requires_grad=False
                )

            # Return the modified model
            return model
        else:
            # For standard problem types, load the model without additional customizations
            return AutoModelForSequenceClassification.from_pretrained(
                dir_, num_labels=num_labels, problem_type=problem_type
            )

    return reload_func  # Return the nested function for later use

# Define the list of problem types requiring alternative loss handling
ALTERNATIVE_LOSSES: List[str] = ['soft_single_label_classification', 'max_exponential_margin']


# Define the custom loss functions for alternative problem types
def calculate_soft_single_label_classifiction(labels, logits, num_labels):
    loss_fct = torch.nn.CrossEntropyLoss()
    if num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
        return loss_fct(logits.view(-1, num_labels), labels.view(-1))
    else:
        return loss_fct(logits.view(-1, num_labels), labels.view(-1, num_labels))

def calculate_exponential_loss(labels, logits):
    z_c = logits[torch.arange(len(labels)), labels]
    z_c = torch.reshape(z_c, (len(labels), 1))
    z_c = z_c.repeat(1, logits.shape[1])
    loss = torch.exp(logits - z_c)
    loss_m = torch.mean(loss) - 1.0
    return loss_m

def alternative_loss_decorator(model, forward_func, problem_type='single_label_classification'):
    """
    Implement alternative loss as a wrapper to enable usage
    under HF for various base models

    Args:
        model: model
        forward_func: the forward
        problem_type: the name for the alternative loss

    Returns:

    """
    # doesn't currently work for multiple GPU training under HF
    num_labels = model.num_labels
    def inner_forward(*args, **kwargs):
        # storing time before function execution

        labels = None
        if 'labels' in kwargs and problem_type in ALTERNATIVE_LOSSES:
            labels = kwargs.pop('labels')
        #output is a dictionary and class--need to set key for HF trainer
        output = forward_func(*args, **kwargs)
        logits = output.logits

        if problem_type == "soft_single_label_classification" and labels is not None:
            output['loss'] = calculate_soft_single_label_classifiction(labels, logits, model.num_labels)
        elif problem_type == 'max_exponential_margin' and labels is not None:
            # from https://github.com/peiwang062/maxgrad/blob/main/train_chineseChars_maxgrad.py#L302C9-L306C40
            output['loss'] = calculate_exponential_loss(labels, logits)
        if problem_type not in ALTERNATIVE_LOSSES and 'labels' in kwargs:
            assert('loss' in output)

        return output
    return inner_forward



        



    


