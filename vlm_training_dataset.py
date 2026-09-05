import pandas as pd
from torch.utils.data import Dataset

from dataset_pipeline import sample_video_frames


class AHCVLMTrainingDataset(Dataset):
    """
    Converts AHC temporal-window CSV data into Qwen3-VL training examples.

    Each example contains:
        - 8 RGB frames from the temporal window
        - a classification prompt
        - the ground-truth class as the target answer
    """

    def __init__(self, csv_path, num_frames=8):
        self.df = pd.read_csv(csv_path)
        self.num_frames = num_frames

        self.classes = sorted(self.df["label"].unique())

        print(f"Loaded {csv_path}")
        print(f"Total examples: {len(self.df)}")
        print(f"Classes: {self.classes}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        video_path = row["video_path"]
        start_sec = float(row["window_start"])
        end_sec = float(row["window_end"])
        label = row["label"]

        # Extract the temporal frames using our existing pipeline.
        frames = sample_video_frames(
            video_path,
            start_sec,
            end_sec,
            num_frames=self.num_frames
        )

        classes_str = ", ".join(self.classes)

        prompt = (
            "You are an anomaly detection system for traffic and surveillance "
            "video. Analyze this video window and classify it into exactly one "
            f"of these classes: [{classes_str}]. "
            "Output strictly the class name and nothing else."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frames
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": label
                    }
                ]
            }
        ]

        return {
            "messages": messages,
            "frames": frames,
            "label": label,
            "video_path": video_path,
            "start_sec": start_sec,
            "end_sec": end_sec
        }


def test_dataset(csv_path="train_windows.csv"):
    """
    Small sanity test before connecting this dataset to SFTTrainer.
    """

    dataset = AHCVLMTrainingDataset(
        csv_path=csv_path,
        num_frames=8
    )

    print("\n--- VLM Dataset Test ---")

    sample = dataset[0]

    print("Dataset size:", len(dataset))
    print("Number of frames:", len(sample["frames"]))
    print("Frame 0 shape:", sample["frames"][0].shape)
    print("Frame 0 dtype:", sample["frames"][0].dtype)

    print("Label:", sample["label"])
    print("Video:", sample["video_path"])
    print("Start:", sample["start_sec"])
    print("End:", sample["end_sec"])

    print("\nConversation:")
    for message in sample["messages"]:
        print("Role:", message["role"])

        for content in message["content"]:
            if content["type"] == "video":
                print("  Video: 8 frames")
            elif content["type"] == "text":
                print("  Text:", content["text"])

    print("\nVLM dataset test successful.")


if __name__ == "__main__":
    test_dataset()