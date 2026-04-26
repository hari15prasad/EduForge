import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("HF_TOKEN")
repo_id = "Hari15prasad/EduForge"

api = HfApi()

print(f"Uploading BLOG.md to {repo_id}...")
try:
    api.upload_file(
        path_or_fileobj="BLOG.md",
        path_in_repo="BLOG.md",
        repo_id=repo_id,
        repo_type="space",
        token=token
    )
    print("Upload successful!")
except Exception as e:
    print(f"Upload failed: {e}")
