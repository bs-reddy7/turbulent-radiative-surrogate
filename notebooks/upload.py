from huggingface_hub import HfApi
import shutil
import os

api = HfApi()
repo_id = "Sevenzoro321/trl2d-surrogate"

# Copy files to a single folder
os.makedirs("/workspace/upload_folder", exist_ok=True)
shutil.copy(
    "/workspace/the_well/the_well/benchmark/experiments/turbulent_radiative_layer_2D-fno-FNO-0.001/0/checkpoints/best.pt",
    "/workspace/upload_folder/best.pt"
)
shutil.copy(
    "/workspace/turbulent-radiative-surrogate/model_predictions.png",
    "/workspace/upload_folder/model_predictions.png"
)

api.upload_folder(
    folder_path="/workspace/upload_folder",
    repo_id=repo_id,
    commit_message="Add model checkpoint and visualization",
)
print("Upload complete!")