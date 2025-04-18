from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="flan-t5-xl-finetuned-tatqa/final_model",
    repo_id="base/flan-t5-xl_finance_QA",
    repo_type="model",
)
