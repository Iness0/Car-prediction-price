import pandas as pd
from model_utils import TEST_URL

df = pd.read_csv(TEST_URL)
df.sample(20, random_state=42).to_csv("pack/sample_upload.csv", index=False)
print("saved pack/sample_upload.csv", df.shape)
