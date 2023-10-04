import os
import s3fs
import re
import shutil
from src.bucket.cloud_bucket import CloudBucket
from transformers import TrainingArguments

class Storj(CloudBucket):
    def __init__(self, bucket_name: str):
        try:
            ACCESS_KEY_ID = os.environ["ACCESS_KEY_ID"]
            SECRET_ACCESS_KEY = os.environ["SECRET_ACCESS_KEY"]
            ENDPOINT_URL = os.environ["ENDPOINT_URL"]
        except KeyError:
            raise Exception(
                "Need to pass in Storj credentials as environment variables"
            )

        storage_options = {
            "key": ACCESS_KEY_ID,
            "secret": SECRET_ACCESS_KEY,
            "client_kwargs": {"endpoint_url": ENDPOINT_URL},
        }
        self.s3 = s3fs.S3FileSystem(**storage_options)
        self.bucket_name = bucket_name

    def get_job_directory(self, job_id: str):
        return f"{self.bucket_name}/training-job-id-{job_id}/"

    def filter_ckpt_folders(self, ckpt_folders: list, job_id: str):
        directory_pattern = rf"^{self.get_job_directory(job_id)}checkpoint-(\w+)$"
        max_checkpoint = -1
        return_folder = ""
        for folder in ckpt_folders:
            match = re.match(directory_pattern, folder)
            ## If final checkpoint, return immediately
            if match and match.group(1) == "final":
                return folder, match.group(1)
            if match and (int(match.group(1)) > max_checkpoint):
                max_checkpoint = int(match.group(1))
                return_folder = folder
        return return_folder, max_checkpoint

    def pull_checkpoints_from_cloud(self, training_args: TrainingArguments) -> bool:
        """
        checks the cloud bucket for checkpoints
        if exists, it downloads locally to the specified output_dir and returns true
        else, it returns false
        """
        try:
            ckpt_folders = self.s3.ls(
                f"s3://{self.get_job_directory(training_args.job_id)}"
            )
        except FileNotFoundError:
            print(
                f"No checkpoints exist in Storj bucket demo-bucket for job id={training_args.job_id}"
            )
            return (False, False)

        ckpt_folder, step = self.filter_ckpt_folders(
            ckpt_folders, training_args.job_id
        )
        if ckpt_folder:
            local_dir = os.path.join(training_args.output_dir, f"checkpoint-{step}")
            self.s3.get(
                f"s3://{ckpt_folder}", local_dir, recursive=True
            )  # download all files
            print(f"Checkpoint from step {step} successfully downloaded!")
            return (True, step == "final")
        # the directory /{bucket-name}/training-job-id{job_id}/ exists but no valid checkpoint folders
        else:
            print(
                f"No valid checkpoint directories exist in Storj bucket demo-bucket for job id={training_args.job_id}"
            )
            return (False, False)

    def save_checkpoints_to_cloud(self, output_dir: str, step: str, job_id: str):
        local_ckpt_dir = f"{output_dir}/checkpoint-{step}"
        for filename in os.listdir(local_ckpt_dir):
            bucket_path = (
                f"s3://{self.get_job_directory(job_id)}checkpoint-{step}/{filename}"
            )
            try:
                with self.s3.open(bucket_path, "wb") as f:
                    f.write(open(os.path.join(local_ckpt_dir, filename), "rb").read())
            except FileNotFoundError as e:
                raise Exception(
                    f"Bucket {self.bucket_name} does not exist on user Storj account"
                )

        # remove local files after upload except for final ckpt which is needed for inference script
        if step != "final":
            shutil.rmtree(local_ckpt_dir)
