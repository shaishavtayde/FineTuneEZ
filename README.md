# 🚀 FineTuneEZ: Simplifying Model Training & Few-Shot Learning 🧠✨

FineTuneEZ is your ultimate tool for training, fine-tuning, and few-shot learning with Hugging Face pretrained models and LLMs. Designed for data scientists and ML engineers, FineTuneEZ eliminates boilerplate, accelerates workflows, and makes complex tasks a breeze with simple, script-driven execution.

---

## 🌟 Why FineTuneEZ?

- **No Boilerplate, No Hassle**: Forget repetitive starter code—focus on what matters!
- **Plug-and-Play**: Create a script with your parameters, and let FineTuneEZ handle the rest.
- **Few-Shot Learning**: Effortlessly implement and experiment with few-shot prompt engineering.
- **Versatile**: From fine-tuning transformers to downstream NLP tasks, we've got you covered.
- **Hugging Face Integrated**: Built on the trusted Hugging Face ecosystem.

---

## 🔥 Features

- **Streamlined Training**: Train or fine-tune Hugging Face models with minimal setup.
- **Few-Shot Learning Made Easy**: Optimize your prompts and evaluate results effortlessly.
- **Customizable & Scalable**: Supports small datasets, large-scale tasks, and everything in between.
- **Script-Driven Execution**: Run training sessions using simple scripts—no coding required.
- **Support for Pretrained Models**: Leverage state-of-the-art transformers and LLMs.

---
### **Training Arguments**

FineTuneEZ supports a wide range of arguments for full control over your training and evaluation workflows. Below is the detailed list:

---

#### **Common Arguments**

| Argument                | Type         | Default                     | Description                                                                                      |
|-------------------------|--------------|-----------------------------|--------------------------------------------------------------------------------------------------|
| `--tokenizer`           | `str` or callable | `None`                      | Name or directory of the Hugging Face tokenizer to be used.                                     |
| `--model`               | `str`       | _(Required)_                | Name or path to the Hugging Face model to be fine-tuned.                                         |
| `--output-dir`          | `str`       | _(Required)_                | Directory to save the trained model and outputs.                                                |
| `--num-labels`          | `int`       | `2`                         | Number of labels for the classification task (e.g., `2` for binary classification).             |
| `--problem-type`        | `str`       | `'single_label_classification'` | Type of problem: `'regression'`, `'single_label_classification'`, `'multi_label_classification'`, or custom alternatives from `utils.ALTERNATIVE_LOSSES`. |
| `--lr`                  | `float`     | `2e-5`                      | Learning rate for training.                                                                     |
| `--optimizer`           | callable    | `'adamw'`                   | Optimizer to use for training, such as AdamW (default).                                          |
| `--validation-data`     | callable    | _(Required)_                | Path to the validation dataset directory, loaded via `load_from_disk`.                          |
| `--test-data`           | callable    | `None`                      | Path to the test dataset directory (optional).                                                  |
| `--lr-plateau-patience` | `int`       | `None`                      | Number of epochs to wait before triggering early stopping based on evaluation F1 score.         |
| `--lr-plateau-factor`   | `float`     | `0.1`                       | Factor by which the learning rate will be reduced when patience is reached.                     |
| `--batch-size`          | `int`       | _(Required)_                | Number of samples in each training batch.                                                       |

---

#### **Training-Specific Arguments**

| Argument                | Type         | Default                     | Description                                                                                      |
|-------------------------|--------------|-----------------------------|--------------------------------------------------------------------------------------------------|
| `--labeled-data`        | callable     | _(Required)_                | Path to the labeled training dataset directory, loaded via `load_from_disk`.                    |
| `--max-training-epochs` | `int`        | `32`                        | Maximum number of epochs to train the model.                                                    |
| `--save-initialization` | `flag`       | `False`                     | Save the model’s random initialization before training (useful for reproducibility).            |
| `--reweight`            | `flag`       | `False`                     | Reweight classes to handle class imbalance during training.                                      |


## 🛠️ How It Works

1. Clone the repository
2. python setup.py install
3. Head over scripts/{write your own script} or (use the given)
4. Run in terminal



