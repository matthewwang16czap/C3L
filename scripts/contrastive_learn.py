import torch
import torch.nn.functional as F
import torch.nn as nn
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
from utils import tokenize_input
import pandas as pd
import bitsandbytes as bnb


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
            qa = f"{conv.roles[0]}: {question} {conv.roles[1]}: {answer}"
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

        anchor_id = tokenize_input(
            instruction,
            False,
            self.tokenizer,
            self.model_config.mm_use_im_start_end,
            conv,
        )

        return {
            "anchor_id": anchor_id,
            "positive_id": positive_id,
            "negative_ids": negative_ids,
        }


def make_contrastive_collate_fn(pad_token_id):
    def collate_fn(batch):
        anchor_ids = [item["anchor_id"] for item in batch]
        positive_ids = [item["positive_id"] for item in batch]
        negative_idss = [item["negative_ids"] for item in batch]

        num_negs = len(negative_idss[0])
        grouped_negatives = [
            [sample[i] for sample in negative_idss] for i in range(num_negs)
        ]

        return {
            "anchor_ids": torch.nn.utils.rnn.pad_sequence(
                anchor_ids, batch_first=True, padding_value=pad_token_id
            ),
            "positive_ids": torch.nn.utils.rnn.pad_sequence(
                positive_ids, batch_first=True, padding_value=pad_token_id
            ),
            "negative_idss": [
                torch.nn.utils.rnn.pad_sequence(
                    group, batch_first=True, padding_value=pad_token_id
                )
                for group in grouped_negatives
            ],
        }

    return collate_fn


class AffineTransformationLayer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
        )
        nn.init.xavier_normal_(self.proj[0].weight)  # Xavier normal initialization
        nn.init.zeros_(self.proj[0].bias)  # Initialize biases to zeros

    def forward(self, x, mask=None):
        # x: (B, L, D) where L = seq_len, D = embed_dim
        x = self.proj(x)  # (B, L, D)
        if mask is not None:
            # mask: (B, L), 1 = valid, 0 = pad
            mask = mask.unsqueeze(-1)  # (B, L, 1)
            x = x * mask  # zero out padded embeddings
            sum_x = x.sum(dim=1)  # (B, D)
            count = mask.sum(dim=1).clamp(min=1e-6)  # (B, 1)
            count = count.masked_fill(count == 0, 1)
            return sum_x / count  # (B, D)
        else:
            return x.mean(dim=1)  # (B, D) if no mask


def contrastive_loss(anchor, positive, negatives, temperature=0.2):
    # anchor: (B,1,D), positive: (B,1,D), negatives: (B,N,D)
    sims_pos = F.cosine_similarity(anchor, positive, dim=-1)  # (B,1)
    sims_neg = F.cosine_similarity(anchor, negatives, dim=-1)  # (B,N)
    logits = torch.cat([sims_pos, sims_neg], dim=1) / temperature  # (B, 1+N)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


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
        use_flash_attn=args.use_flash_attn,
        offload_folder="./offload",
    )

    # Freeze vision tower parameters
    # for param in model.get_vision_tower().parameters():
    #     param.requires_grad = False
    # model.get_vision_tower().eval()
    model.train()

    text_proj = AffineTransformationLayer(embed_dim=4096).to(
        dtype=model.model.embed_tokens.weight.dtype, device=args.device
    )

    optimizer = bnb.optim.Adam8bit(
        list(model.model.embed_tokens.parameters()) + list(text_proj.parameters()),
        lr=1e-5,
        betas=(0.9, 0.95),
    )

    # Create dataset and dataloader
    dataset = ContrastiveDataset(
        args.question_file, args.i2c_file, model.config, tokenizer, args.conv_mode
    )
    collate_fn = make_contrastive_collate_fn(tokenizer.pad_token_id)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Training loop
    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(dataloader):
            anchor_ids = batch["anchor_ids"]
            positive_ids = batch["positive_ids"]
            negative_idss = batch["negative_idss"]
            anchor_embed = model.model.embed_tokens(
                anchor_ids.to(args.device)
            )  # (B, L, D)
            anchor_mask = (anchor_ids != tokenizer.pad_token_id).float()  # (B, L)
            anchor_embedding = text_proj(
                anchor_embed.to(args.device), mask=anchor_mask.to(args.device)
            ).unsqueeze(
                1
            )  # (B, 1, D)
            positive_embed = model.model.embed_tokens(
                positive_ids.to(args.device)
            )  # (B, L, D)
            positive_mask = (positive_ids != tokenizer.pad_token_id).float()  # (B, L)
            positive_embedding = text_proj(
                positive_embed.to(args.device), mask=positive_mask.to(args.device)
            ).unsqueeze(
                1
            )  # (B, 1, D)
            negative_embeddings = []
            for negative_ids in negative_idss:
                negative_embed = model.model.embed_tokens(
                    negative_ids.to(args.device)
                )  # (B, L, D)
                megative_mask = (
                    negative_ids != tokenizer.pad_token_id
                ).float()  # (B, L)
                negative_embeddings.append(
                    text_proj(
                        negative_embed.to(args.device),
                        mask=megative_mask.to(args.device),
                    )
                )  # (B, D)
            negative_embeddings = torch.stack(negative_embeddings, dim=1)
            loss = contrastive_loss(
                anchor_embedding,
                positive_embedding,
                negative_embeddings,
                temperature=args.temperature,
            )

            optimizer.zero_grad()  # Clear any previous gradients
            loss.backward()  # Compute gradients
            optimizer.step()  # Update parameters

            # print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")
        print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Save the model (including the fine-tuned weights)
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    # Save the affine layer separately
    torch.save(text_proj.state_dict(), args.output_dir + "/affine_layer.pth")


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
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints/llava-v1.5-7b-contrastive-learned",
    )
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--load-4bit", type=bool, default=False)
    parser.add_argument("--use-flash-attn", type=bool, default=False)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    contrastive_learning_train(args)
