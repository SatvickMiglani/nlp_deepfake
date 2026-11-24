from huggingface_hub import HfApi
api = HfApi()

print("Searching for all datasets with fakeavceleb...")
res = list(api.list_datasets(search="fakeavceleb", limit=50))
for d in res:
    print(f"Dataset: {d.id} | Downloads: {d.downloads} | Gated: {getattr(d, 'gated', False)}")

print("\nSearching for all models with fakeavceleb...")
models = list(api.list_models(search="fakeavceleb", limit=20))
for m in models:
    print(f"Model: {m.id} | Downloads: {m.downloads}")
