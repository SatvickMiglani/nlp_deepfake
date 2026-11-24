from huggingface_hub import HfApi
api = HfApi()

for repo in ['bitmind/FaceForensicsC23', 'gonnerthetooner/FaceForensics']:
    print(f'=== {repo} ===')
    try:
        files = list(api.list_repo_tree(repo, repo_type='dataset', recursive=False))
        for f in files[:10]:
            print(f'  {f.path} ({getattr(f, "size", None)})')
    except Exception as e:
        print(f'  Failed: {e}')
