from huggingface_hub import HfApi
api = HfApi()

print("Listing val top-level:")
files = list(api.list_repo_tree("cocothenm/unzip_avdeepfake", repo_type="dataset", path_in_repo="val", recursive=False))
for f in files:
    print(f.path)
