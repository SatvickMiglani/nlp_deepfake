from huggingface_hub import HfApi
api = HfApi()

queries = ["deepfake", "faceforensics", "celeb-df", "dfdc", "fake audio video", "multimodal deepfake"]
found_datasets = {}

for q in queries:
    ds_list = list(api.list_datasets(search=q, limit=20))
    for d in ds_list:
        if d.id not in found_datasets:
            gated = getattr(d, 'gated', False)
            found_datasets[d.id] = {
                'downloads': d.downloads,
                'gated': gated,
                'likes': getattr(d, 'likes', 0)
            }

print(f"Total found: {len(found_datasets)}")
# Sort by downloads
sorted_ds = sorted(found_datasets.items(), key=lambda x: x[1]['downloads'], reverse=True)
for ds_id, meta in sorted_ds[:30]:
    print(f"{ds_id:50} | Downloads: {meta['downloads']:6} | Gated: {str(meta['gated']):6} | Likes: {meta['likes']:3}")
