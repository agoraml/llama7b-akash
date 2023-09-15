from abc import ABC, abstractmethod
from transformers import TrainingArguments


class CloudBucket(ABC):
    @abstractmethod
    def pull_checkpoints_from_cloud(training_args: TrainingArguments) -> bool:
        pass

    @abstractmethod
    def save_checkpoints_to_cloud(output_dir: str, step_num: int, job_id: int) -> bool:
        pass
