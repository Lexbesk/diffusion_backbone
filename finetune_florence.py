from glob import glob
from torch.utils.data import DataLoader
from tqdm import tqdm 
from transformers import AdamW, get_scheduler
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from torch.utils.data import Dataset 
import json
from PIL import Image
import wandb
from itertools import cycle
import math


# Initialize wandb
wandb.init(
    project="florence_finetuning",  # Replace with your project name
    name="tower3_only",
    config={
        "model": "microsoft/Florence-2-base-ft",
        "epochs": 100,
        "batch_size": 6,
        "learning_rate": 1e-6,
        "optimizer": "AdamW",
        "scheduler": "linear",
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-base-ft",
    trust_remote_code=True,
    revision='refs/pr/6'
).to(device)
processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-base-ft",
    trust_remote_code=True, revision='refs/pr/6'
)
# for param in model.vision_tower.parameters():
#     param.is_trainable = False


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
        # Load image
        image = Image.open(self.annotations['images'][idx])

        # Load question (input text)
        question = "<ROB> What should the robot do to " + self.annotations['description'][idx] + "?"

        # Load answer (output text)
        answer = (
            f"{self.annotations['subgoals'][idx]}"
            # + f"{self._objdict2str(self.annotations['objects'][idx])} "
            # + f"trajectory{self._gripper2str(self.annotations['gripper_trace'][idx])}"
        )
        return question, answer, image

    @staticmethod
    def _objdict2str(obj):
        formatted_items = []
        for key, val in obj.items():
            formatted_vals = ''.join(f"<loc_{str(_i)[:4]}>" for _i in val)
            formatted_item = f"{key}{formatted_vals}"
            formatted_items.append(formatted_item)
        return ' '.join(formatted_items)

    @staticmethod
    def _gripper2str(gripper):
        # get 10 evenly spaced indices from gripper
        # print(len(gripper))
        half_len = math.ceil(len(gripper)/2)
        if half_len > 6:
            gripper_dex = list(range(0, half_len, math.ceil(half_len / 6)))

            if half_len not in gripper_dex:
                # replace the last index with len(gripper)//2
                gripper_dex[-1] = half_len

            while len(gripper_dex) < 6:
                # get diff of list elements
                diff = [gripper_dex[i+1] - gripper_dex[i] for i in range(len(gripper_dex)-1)]
                # get index of max diff
                max_diff = diff.index(max(diff))
                # get_value of max diff
                max_diff_val = diff[max_diff]
                # insert new value
                new_val = max_diff_val // 2 + gripper_dex[max_diff]
                gripper_dex.insert(max_diff+1, new_val)

            # print(gripper_dex)
            # for every index in gripper dex add i and i+1 to the list
            new_gripper = [gripper[2*i:2*i+2] for i in gripper_dex]

            gripper = []
            for item in new_gripper:
                # if item is list
                if isinstance(item, list):
                    gripper.extend(item)
                else:
                    gripper.append(item)

        formatted_elements = ''.join(f"<loc_{str(_i)[:4]}>" for _i in gripper)
        # print(formatted_elements)
        return formatted_elements


def collate_fn(batch): 
    questions, answers, images = zip(*batch)
    inputs = processor(
        text=list(questions),
        images=list(images),
        return_tensors="pt",
        padding=True
    )  # Keep data on CPU for now
    return inputs, answers


# Define how many steps constitute an epoch
steps_per_epoch = 500
epochs = 100  # Each of these "epochs" is now 500 steps

# Create your datasets and dataloaders
train_dataset = RLBenchDataset(split='train', seed=42, split_ratio=0.9)
val_dataset = RLBenchDataset(split='val', seed=42, split_ratio=0.9)

batch_size = 4
num_workers = 0

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, 
    collate_fn=collate_fn, num_workers=num_workers, shuffle=True
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, 
    collate_fn=collate_fn, num_workers=num_workers
)

# Calculate total number of training steps
num_training_steps = epochs * steps_per_epoch

optimizer = AdamW(model.parameters(), lr=1e-6)
lr_scheduler = get_scheduler(
    name="linear", 
    optimizer=optimizer, 
    num_warmup_steps=0, 
    num_training_steps=num_training_steps,
)

losses = []

# Use cycle to continuously draw batches from train_loader
train_iterator = cycle(train_loader)

for epoch in range(epochs): 
    model.train() 
    train_loss = 0

    # Run exactly 500 steps per epoch
    for step in tqdm(range(steps_per_epoch), desc=f"Training Epoch {epoch + 1}/{epochs}"):
        inputs, answers = next(train_iterator)
        input_ids = inputs["input_ids"].to(device)
        pixel_values = inputs["pixel_values"].to(device)
        labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True, return_token_type_ids=False).input_ids.to(device)

        outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        train_loss += loss.item()

        del inputs, answers, labels, outputs, loss

    avg_train_loss = train_loss / steps_per_epoch
    wandb.log({"train_loss": avg_train_loss})
    print(f"Average Training Loss (Epoch {epoch+1}): {avg_train_loss}")

    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f"new_florence_finetuned_{epoch + 1}_epoch.pth")

    # Validation after every "epoch" of 500 steps
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, answers in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):
            input_ids = inputs["input_ids"].to(device)
            pixel_values = inputs["pixel_values"].to(device)
            labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True, return_token_type_ids=False).input_ids.to(device)
            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()

            del inputs, answers, labels, outputs, loss

    avg_val_loss = val_loss / len(val_loader)
    wandb.log({"val_loss": avg_val_loss})
    print(f"Average Validation Loss (Epoch {epoch+1}): {avg_val_loss}")
