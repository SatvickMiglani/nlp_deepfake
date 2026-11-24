from huggingface_hub import HfApi
api = HfApi()

print("=== cocothenm/unzip_avdeepfake/val ===")
try:
    files = list(api.list_repo_tree("cocothenm/unzip_avdeepfake", repo_type="dataset", path_in_repo="val", recursive=True))
    print(f"Total files in val: {len(files)}")
    for f in files[:20]:
        print(f"  {f.path} ({getattr(f, 'size', None)})")
except Exception as e:
    print("Failed:", e)

print("\n=== UniDataPro/deepfake-videos-dataset ===")
try:
    files = list(api.list_repo_tree("UniDataPro/deepfake-videos-dataset", repo_type="dataset", path_in_repo="video", recursive=True))
    print(f"Total files in video: {len(files)}")
    for f in files[:20]:
        print(f"  {f.path} ({getattr(f, 'size', None)})")
except Exception as e:
    print("Failed:", e)

print("\n=== angads24/deepfake-video ===")
try:
    files = list(api.list_repo_tree("angads24/deepfake-video", repo_type="dataset", recursive=True))
    print(f"Total files: {len(files)}")
    for f in files[:20]:
        print(f"  {f.path} ({getattr(f, 'size', None)})")
except Exception as e:
    print("Failed:", e)
