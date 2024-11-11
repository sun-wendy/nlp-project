## Installation

```
python -m venv stats
source stats/bin/activate
pip install -r requirements.txt
```

When you want to update packages in your environment, run:
```
pip freeze > requirements.txt
```

<br>

## Set up Ollama for inference 🦙

1. Download Ollama app [here](https://ollama.com/).
2. Open the app.
3. Run `pip install ollama` in terminal.
4. If you want to run model e.g. `llama3.2`, run `ollama run llama3.2` in terminal (this will load the model).
5. Open `run_ollama.ipynb`, run the code (make sure to change the model name in second cell).
