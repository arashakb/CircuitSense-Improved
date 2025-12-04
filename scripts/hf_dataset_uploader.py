import os
from huggingface_hub import HfApi

# Read token from environment (recommended: export HF_TOKEN in your shell)
hf_token = os.environ.get("HF_TOKEN")

# Initialize the API with token (no token kwarg on upload_large_folder)
api = HfApi(token=hf_token)

# Upload entire folder using the large-folder helper
api.upload_large_folder(
    folder_path="datasets/symbolic_level15_27",
    repo_id="arashakb/Circuitsense-6k",
    repo_type="dataset",
)