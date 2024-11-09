from datasets import load_dataset
import pandas as pd


dataset = load_dataset("cais/mmlu", "high_school_chemistry")
df = pd.DataFrame(dataset['test'])

# Display the DataFrame
print(df)