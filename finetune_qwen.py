from glob import glob
from torch.utils.data import DataLoader
from tqdm import tqdm 
from transformers import AdamW, get_scheduler
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from torch.utils.data import Dataset 
import json
from PIL import Image
import wandb
from itertools import cycle
from qwen_vl_utils import process_vision_info


SYSTEM_MSG = """You are a Vision Language Model specialized in interpreting visual data from tabletop robot manipulation scenes.
Your task is to analyze the provided front camera image and respond to queries with concise answers, usually a short phrase.
The images show a tabletop robot manipulation setup, with a single franka robot arm and different objects that need to be rearranged.
The input additionally contains a language instruction that serves as a high-level task description.
The output is the next subgoal that the robot arm should achieve in order to perform the task indicated by the input instruction.
Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation."""


# Initialize wandb
wandb.init(
    project="qwen_finetuning",  # Replace with your project name
    name="tower3_only",
    config={
        "model": "Qwen2-VL-2B-Instruct",
        "epochs": 100,
        "batch_size": 6,
        "learning_rate": 1e-6,
        "optimizer": "AdamW",
        "scheduler": "linear",
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    for name, param in model.named_parameters():
        if not name.startswith('model'):
            param.requires_grad = False
    processor = Qwen2VLProcessor.from_pretrained(model_id)
    return model, processor


class RLBenchDataset(Dataset):

    def __init__(self, 
                 root='/home/mbronars/workspace/scripts/VLA_comp_manip/', 
                 split='train', 
                 seed=42, 
                 split_ratio=0.8):
        """
        Args:
            root (str): Path to the directory containing the JSON annotation file(s).
            split (str): 'train' or 'val' to specify which split of the data to return.
            seed (int): Random seed for reproducible splits.
            split_ratio (float): Ratio of training samples (e.g., 0.8 for 80% train, 20% val).
        """

        # Validate the split argument
        assert split in ['train', 'val'], "split must be either 'train' or 'val'"

        # Load and merge all annotations
        all_annos = {
            'images': [],
            'objects': [],
            'description': [],
            'subgoals': [],
            'gripper_trace': []
        }

        extend_keys = ['objects', 'subgoals', 'gripper_trace']

        # You might need to adjust this glob if you have multiple json files
        annotation_files = glob("/data/group_data/katefgroup/VLA/data/jsons/tower3_full_annotations.json") #f'{root}/new_florence_annotations.json')

        for fname in annotation_files:
            with open(fname, 'r') as fid:
                _annos = json.load(fid)
            all_demos = _annos.keys()
            # sort by demo number where name is 'name_X'
            all_demos = sorted(all_demos, key=lambda x: int(x.split('_')[-1]))
            split_point = int(split_ratio * len(all_demos))
            if split == 'train':
                target_demos = all_demos[:split_point]
            else:
                target_demos = all_demos[split_point:]
            for task in target_demos:
                for key in extend_keys:
                    all_annos[key].extend(_annos[task][key])
                all_annos['description'].extend([_annos[task]['description']] * len(_annos[task]['images']))
                all_annos['images'].extend([
                    f'{_annos[task]["image_path"]}/{_n}' for _n in _annos[task]['images']
                ])

        self.annotations = {
            'images': all_annos['images'],
            'objects': all_annos['objects'],
            'description': all_annos['description'],
            'subgoals': all_annos['subgoals'],
            'gripper_trace': all_annos['gripper_trace'],
        }

    def __len__(self):
        return len(self.annotations['images'])
    
    def __getitem__(self, idx):
        return {"messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_MSG}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": Image.open(self.annotations['images'][idx])
                    },
                    {
                        "type": "text",
                        "text": "What should the robot do to " + self.annotations['description'][idx] + "?"
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": f"{self.annotations['subgoals'][idx]}"
                    }
                ]
            }
        ]}


def find_assistant_content_sublist_indexes(l):
    start_indexes = []
    end_indexes = []

    # Iterate through the list to find starting points
    for i in range(len(l) - 2):
        # Check if the current and next elements form the start sequence
        if l[i] == 151644 and l[i+1] == 77091 and l[i+2] == 198:
            start_indexes.append(i+3)
            # Now look for the first 151645 and 198 after the start
            for j in range(i+3, len(l)-1):
                if l[j] == 151645 and l[j+1] == 198:
                    end_indexes.append(j+2)
                    break  # Move to the next start after finding the end

    return list(zip(start_indexes, end_indexes))


def collate_fn(batch):
    return [item["messages"] for item in batch]


def main():
    # Define how many steps constitute an epoch
    steps_per_epoch = 500
    epochs = 100  # Each of these "epochs" is now 500 steps
    # Calculate total number of training steps
    num_training_steps = epochs * steps_per_epoch

    # Create datasets and dataloaders
    train_dataset = RLBenchDataset(split='train', seed=42, split_ratio=0.9)
    val_dataset = RLBenchDataset(split='val', seed=42, split_ratio=0.9)

    batch_size = 1
    num_workers = 0
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        collate_fn=collate_fn, num_workers=num_workers, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, 
        collate_fn=collate_fn, num_workers=num_workers
    )

    # Load model
    model, processor = load_model()
    optimizer = AdamW(model.parameters(), lr=1e-6)
    lr_scheduler = get_scheduler(
        name="linear", 
        optimizer=optimizer, 
        num_warmup_steps=0, 
        num_training_steps=num_training_steps,
    )

    # Use cycle to continuously draw batches from train_loader
    train_iterator = cycle(train_loader)

    # Training loop
    for epoch in range(epochs): 
        model.train() 
        train_loss = 0

        # Run exactly 500 steps per epoch
        for _ in tqdm(range(steps_per_epoch), desc=f"Training Epoch {epoch + 1}/{epochs}"):
            # Load conversations
            messages = next(train_iterator)
            # Convert using template
            texts = [
                processor.apply_chat_template(
                    msg,
                    tokenize=False,
                    add_generation_prompt=False
                )
                for msg in messages
            ]
            # Process images
            image_inputs, _ = process_vision_info(messages)
            # Tokenize
            inputs = processor(
                text=texts,
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            ).to(device)
            labels_list = []
            for ids_list in inputs['input_ids']:
                label_ids = [-100] * len(ids_list)
                for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
                    label_ids[begin_end_indexs[0]:begin_end_indexs[1]] = ids_list[begin_end_indexs[0]:begin_end_indexs[1]]
                labels_list.append(label_ids)
            labels_ids = torch.tensor(labels_list, dtype=torch.int64)
            # from ipdb import set_trace as st; st()

            # Forward pass
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(**inputs, labels=labels_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            train_loss += loss.item()

            del inputs, label_ids, outputs, loss

        avg_train_loss = train_loss / steps_per_epoch
        wandb.log({"train_loss": avg_train_loss})
        print(f"Average Training Loss (Epoch {epoch+1}): {avg_train_loss}")

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"new_florence_finetuned_{epoch + 1}_epoch.pth")

        # Validation after every "epoch" of 500 steps
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for messages in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):
                # Convert using template
                texts = [
                    processor.apply_chat_template(
                        msg,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    for msg in messages
                ]
                # Process images
                image_inputs, _ = process_vision_info(messages)
                # Tokenize
                inputs = processor(
                    text=texts,
                    images=image_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                labels_list = []
                for ids_list in inputs['input_ids']:
                    label_ids = [-100] * len(ids_list)
                    for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
                        label_ids[begin_end_indexs[0]:begin_end_indexs[1]] = ids_list[begin_end_indexs[0]:begin_end_indexs[1]]
                    labels_list.append(label_ids)
                labels_ids = torch.tensor(labels_list, dtype=torch.int64)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model(**inputs, labels=labels_ids)
                loss = outputs.loss
                val_loss += loss.item()

                del inputs, label_ids, outputs, loss

        avg_val_loss = val_loss / len(val_loader)
        wandb.log({"val_loss": avg_val_loss})
        print(f"Average Validation Loss (Epoch {epoch+1}): {avg_val_loss}")


if __name__ == "__main__":
    main()
