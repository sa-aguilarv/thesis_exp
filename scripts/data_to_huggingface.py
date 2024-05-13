import os
from huggingface_hub import HfApi

# Get the username and password from environment variables
username = os.getenv("HF_USERNAME")
password = os.getenv("HF_PASSWORD")

# Authenticate with the Hugging Face API
api = HfApi()
api.login(username, password)

# Now you can upload your folder
api.upload_folder(
    folder_path="/media/sophie/dataspace/MBCI/thesis_exp/data/comm_use_subset/pdf_json",
    repo_id="sa-aguilarv/thesis_exp",
    repo_type="dataset",
)