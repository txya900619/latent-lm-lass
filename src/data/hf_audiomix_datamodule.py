from typing import Any

import torch
import torchaudio
from datasets import load_dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class AudioMixDataset(Dataset):
    """Dataset that mixes audio samples during __getitem__."""

    def __init__(
        self,
        dataset: Dataset,
        min_snr: float = 0.0,
        max_snr: float = 0.0,
        is_train: bool = False,
    ) -> None:
        """Initialize the dataset.

        :param dataset: The base dataset to mix samples from
        :param min_snr: Minimum Signal-to-Noise Ratio in dB for mixing
        :param max_snr: Maximum Signal-to-Noise Ratio in dB for mixing
        :param is_train: Whether this is the training dataset
        """
        self.dataset = dataset
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.is_train = is_train
        self.add_noise = torchaudio.transforms.AddNoise()

        # Get youtube_id and start_time lists for faster comparison
        self.youtube_ids = self.dataset["youtube_id"]
        self.start_times = self.dataset["start_time"]

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.dataset)

    def _get_random_noise(self, exclude_sample: dict, exclude_idx: int) -> dict:
        """Get a random noise sample, excluding the current sample.

        :param exclude_sample: The sample to exclude
        :param exclude_idx: The index of the sample to exclude
        :return: A random noise sample
        """
        if self.is_train:
            # For training set, just exclude by index
            available_indices = list(range(len(self.dataset)))
            available_indices.remove(exclude_idx)
            noise_idx = available_indices[torch.randperm(len(available_indices))[0].item()]
            return self.dataset[noise_idx]
        else:
            # For validation/test sets, exclude by youtube_id and start_time
            exclude_youtube_id = exclude_sample["youtube_id"]
            exclude_start_time = exclude_sample["start_time"]

            # Get indices where either youtube_id or start_time is different
            available_indices = [
                i
                for i in range(len(self.dataset))
                if self.youtube_ids[i] != exclude_youtube_id
                or self.start_times[i] != exclude_start_time
            ]

            if not available_indices:
                raise ValueError("No available samples for mixing")

            noise_idx = available_indices[torch.randperm(len(available_indices))[0].item()]
            return self.dataset[noise_idx]

    def __getitem__(self, idx: int) -> dict:
        """Get a mixed audio sample.

        :param idx: Index of the sample to get
        :return: Mixed audio sample
        """
        # Get the clean sample
        clean_sample = self.dataset[idx]

        # Get a random noise sample
        noise_sample = self._get_random_noise(clean_sample, idx)

        # Ensure audio tensors have the same shape
        if clean_sample["audio"]["array"].shape != noise_sample["audio"]["array"].shape:
            # Pad or truncate to match the longer one
            max_len = max(
                clean_sample["audio"]["array"].shape[-1],
                noise_sample["audio"]["array"].shape[-1],
                160000,
            )
            if clean_sample["audio"]["array"].shape[-1] < max_len:
                clean_sample["audio"]["array"] = torch.nn.functional.pad(
                    clean_sample["audio"]["array"],
                    (0, max_len - clean_sample["audio"]["array"].shape[-1]),
                )
            if noise_sample["audio"]["array"].shape[-1] < max_len:
                noise_sample["audio"]["array"] = torch.nn.functional.pad(
                    noise_sample["audio"]["array"],
                    (0, max_len - noise_sample["audio"]["array"].shape[-1]),
                )

        # Add batch dimension if needed
        if len(clean_sample["audio"]["array"].shape) == 1:
            clean_sample["audio"]["array"] = clean_sample["audio"]["array"].unsqueeze(0)
        if len(noise_sample["audio"]["array"].shape) == 1:
            noise_sample["audio"]["array"] = noise_sample["audio"]["array"].unsqueeze(0)

        # Randomly sample SNR from the range
        snr = torch.rand(1) * (self.max_snr - self.min_snr) + self.min_snr

        # Mix the audio using AddNoise
        mixed_audio = self.add_noise(
            clean_sample["audio"]["array"],
            noise_sample["audio"]["array"],
            snr,
        ).squeeze(0)  # Remove batch dimension

        # Create mixed sample
        return {
            "mixed_audio": mixed_audio,
            "target_audio": clean_sample["audio"]["array"],
        }


class HFDataModule(LightningDataModule):
    """`LightningDataModule` for the Huggingface datasets with audio mixing capability."""

    def __init__(
        self,
        dataset_name: str,
        dataset_config: str | None = None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        min_snr: float = 0.0,  # Minimum SNR in dB
        max_snr: float = 0.0,  # Maximum SNR in dB
    ) -> None:
        """Initialize a `HFDataModule`.

        :param dataset_name: Name of the Hugging Face dataset to load
        :param dataset_config: Optional configuration name for the dataset
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        :param min_snr: Minimum Signal-to-Noise Ratio in dB for mixing. Defaults to `0.0`.
        :param max_snr: Maximum Signal-to-Noise Ratio in dB for mixing. Defaults to `0.0`.
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Download data if needed."""
        # Load all splits from Hugging Face
        load_dataset(
            self.hparams.dataset_name,
            self.hparams.dataset_config,
        )

    def setup(self, stage: str | None = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`."""
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        dataset = load_dataset(
            self.hparams.dataset_name,
            self.hparams.dataset_config,
        )
        dataset = dataset.with_format("torch")
        # Create mixed datasets for each split
        self.data_train = AudioMixDataset(
            dataset["train"], self.hparams.min_snr, self.hparams.max_snr, is_train=True
        )
        self.data_val = AudioMixDataset(
            dataset["val"], self.hparams.min_snr, self.hparams.max_snr, is_train=False
        )
        self.data_test = AudioMixDataset(
            dataset["test"], self.hparams.min_snr, self.hparams.max_snr, is_train=False
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader."""
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: str | None = None) -> None:
        """Clean up after training."""
        pass

    def state_dict(self) -> dict[Any, Any]:
        """Called when saving a checkpoint."""
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Called when loading a checkpoint."""
        pass


if __name__ == "__main__":
    _ = HFDataModule(dataset_name="txya900619/audiocaps-16k")
