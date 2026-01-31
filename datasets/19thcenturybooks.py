from datasets import load_dataset

ds = load_dataset("TheBritishLibrary/blbooks", "1500_1899", split="train")

english_ds = ds.filter(lambda x: x["Language_1"] == "English")

print(english_ds)