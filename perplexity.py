import lmppl
import pandas as pd

scorer = lmppl.LM('/mnt/beegfs/agmegias/proyectos/huggingface_models/gpt2', max_length=1024)

# Read data
df = pd.read_json('<PATH>', lines=True)

# Empty dataframe
df_ppl = pd.DataFrame()
ppl_vector = []

for i in df.index:
    text = df['text'][i]
    ppl = scorer.get_perplexity(text, batch=16)
    ppl_vector.append(ppl)

df_ppl['ppl'] = ppl_vector

# Save the perplexity
df_ppl.to_csv('<PATH>', index=False)
