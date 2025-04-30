import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import json
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    get_model_name_from_path,
)
from C3L.scripts.utils import tokenize_input
import pandas as pd


class ContrastiveDataset(Dataset):
    def __init__(self, question_file, i2c_file, model_config, tokenizer, conv_mode):
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.data = []
        with open(question_file, "r") as f:
            for line in f:
                self.data.append(json.loads(line))
        self.I2C = pd.read_csv(i2c_file, header=None)
        self.conv_mode = conv_mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_sample = self.data[idx]
        instruction = data_sample["instruction"]
        conversations = data_sample["conversations"]
        negative_ids = []

        for i in range(0, len(conversations), 2):
            question = conversations[i]["value"]
            answer = conversations[i + 1]["value"]
            conv = conv_templates[self.conv_mode].copy()
            qa = f"{self.conv.roles[0]}: {question} {self.conv.roles[1]}: {answer}"
            input_ids = tokenize_input(
                qa,
                False,
                self.tokenizer,
                self.model_config.mm_use_im_start_end,
                conv,
            )
            negative_ids.append(input_ids)

        positive_idx = self.I2C.iloc[idx].idxmax()
        positive_id = negative_ids[positive_idx]
        negative_ids.pop(positive_idx)
        negative_ids = torch.stack(negative_ids)

        anchor_id = tokenize_input(
            instruction,
            False,
            self.tokenizer,
            self.model_config.mm_use_im_start_end,
            conv,
        )

        return anchor_id, positive_id, negative_ids


def contrastive_loss(anchor, positive, negatives, temperature=0.2):
    """
    Args:
        anchor: [B, 1, D]
        positive: [B, 1, D]
        negatives: [B, N, D]
    Returns:
        Scalar loss (mean over batch)
    """

    # Cosine similarity between anchor and positive → [B, 1]
    pos_sim = F.cosine_similarity(anchor, positive, dim=-1)  # [B, 1]

    # Cosine similarity between anchor and negatives → [B, N]
    neg_sim = F.cosine_similarity(anchor, negatives, dim=-1)  # [B, N]

    # Clamp similarities for stability
    pos_sim = torch.clamp(pos_sim, -1.0, 1.0)
    neg_sim = torch.clamp(neg_sim, -1.0, 1.0)

    # Scale by temperature
    pos_sim = pos_sim / temperature  # [B, 1]
    neg_sim = neg_sim / temperature  # [B, N]

    # Concatenate pos + neg → [B, 1+N]
    sims = torch.cat([pos_sim, neg_sim], dim=1)

    # Compute logsumexp over all similarities
    denom = torch.logsumexp(sims, dim=1)  # [B]
    loss = -pos_sim.squeeze(1) + denom  # [B]

    return loss.mean()


def contrastive_learning_train(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        load_4bit=args.load_4bit,
        use_flash_attn=False,
        offload_folder="./offload",
    )

    # Freeze vision tower parameters
    for param in model.get_vision_tower().parameters():
        param.requires_grad = False
    model.get_vision_tower().eval()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Create dataset and dataloader
    dataset = ContrastiveDataset(
        args.question_file, args.i2c_file, model.config, tokenizer, args.conv_mode
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Training loop
    for epoch in range(args.epochs):
        for batch_idx, (instruction, positive_id, negative_ids) in enumerate(
            dataloader
        ):
            instruction_embedding = model.model.embed_tokens(instruction)
            instruction_embedding = F.normalize(
                instruction_embedding.mean(dim=1), p=2, dim=-1
            )
            positive_id_embedding = model.model.embed_tokens(positive_id)
            positive_id_embedding = F.normalize(
                positive_id_embedding.mean(dim=1), p=2, dim=-1
            )
            negative_ids_embedding = model.model.embed_tokens(negative_ids_embedding)
            negative_ids_embedding = F.normalize(
                negative_ids_embedding.mean(dim=1), p=2, dim=-1
            )
            loss = contrastive_loss(
                instruction_embedding,
                positive_id_embedding,
                negative_ids_embedding,
                temperature=args.temperature,
            )

            optimizer.zero_grad()  # Clear any previous gradients
            loss.backward()  # Compute gradients
            optimizer.step()  # Update parameters

            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=str, default="/home/matthew/models/llava-v1.5-7b"
    )
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument(
        "--question-file", type=str, default="./C3L/data/questions.jsonl"
    )
    parser.add_argument("--i2c-file", type=str, default="./C3L/data/I2C.csv")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--load-4bit", type=bool, default=False)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    contrastive_learning_train(args)
