# Investigating In-Context Curriculum Learning 📚

## Installation 💻

The main results for our project can be replicated from [`notebooks/curriculum.ipynb`](notebooks/curriculum.ipynb), which can be run on Colab with A100 GPU.

To successfully run the notebook:
1. The notebook evaluates Llama-3.2-3B. To successfully load the model, you need a Hugging Face access token, with access to Llama-3 model family.
2. Upload [`prompts.zip`](prompts.zip) from this repository to Colab.
3. The notebook evaluates a non-fine-tuned model, a model fine-tuned without curriculum learning (CL), and a model fine-tuned with CL. You might run into Cuda out of memory error when moving on to fine-tuning the model with CL. If that happens, restart the runtime, and skip fine-tuning model without CL.
