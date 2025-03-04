import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import json
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
import bitsandbytes as bnb  # For 4-bit optimizer


class ContrastiveDataset(Dataset):
    def __init__(
        self, question_file, i2c_path, model, tokenizer, conv_mode, device="cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.data = []
        with open(question_file, "r") as f:
            for line in f:
                self.data.append(json.loads(line))
        self.I2C_tensor = torch.load(i2c_path)
        self.conv = conv_templates[conv_mode].copy()
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_sample = self.data[idx]
        description = data_sample["description"]
        conversations = data_sample["conversations"]
        negatives = []

        for i in range(0, len(conversations), 2):
            question = conversations[i]["value"]
            answer = conversations[i + 1]["value"]
            qa = f"{self.conv.roles[0]}: {question} {self.conv.roles[1]}: {answer}"
            input_ids = self.tokenizer(qa, return_tensors="pt").input_ids.to(
                self.device
            )
            embedding = self.model.model.embed_tokens(input_ids)
            latent_space = F.normalize(embedding.mean(dim=1), p=2, dim=-1)
            negatives.append(latent_space.squeeze(0))

        positive_idx = self.I2C_tensor[idx].argmax()
        positive = negatives[positive_idx]
        negatives.pop(positive_idx)
        negatives = torch.stack(negatives)

        return description, positive, negatives


def contrastive_loss(anchor, positive, negatives, temperature=0.2):
    pos_sim = F.cosine_similarity(anchor.unsqueeze(1), positive, dim=-1)
    neg_sim = F.cosine_similarity(anchor.unsqueeze(1), negatives, dim=-1)

    # Clip cosine similarity to avoid extreme values
    pos_sim = torch.clamp(pos_sim, -1.0, 1.0)
    neg_sim = torch.clamp(neg_sim, -1.0, 1.0)

    pos_sim = pos_sim / temperature
    neg_sim = neg_sim / temperature

    denom = torch.cat([pos_sim, neg_sim], dim=1)
    denom = torch.logsumexp(denom, dim=1)
    loss = -pos_sim + denom

    return loss.mean()


def contrastive_learning_train(args, device="cuda"):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        load_4bit=args.load_4bit,
        output_hidden_states=False,
    )

    # Freeze vision tower parameters
    for param in model.get_vision_tower().parameters():
        param.requires_grad = False
    model.get_vision_tower().eval()
    model.train()

    # Load dataset
    dataset = ContrastiveDataset(
        args.question_file, args.i2c_path, model, tokenizer, args.conv_mode
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=5e-6, betas=(0.9, 0.95))

    for epoch in range(args.epochs):
        for batch_idx, (descriptions, positive, negatives) in enumerate(dataloader):
            optimizer.zero_grad()

            input_ids = tokenizer(
                descriptions, return_tensors="pt", padding=True, truncation=True
            ).input_ids.to(device)
            positive = positive.to(device)
            negatives = negatives.to(device)

            embedding = model.model.embed_tokens(input_ids)
            anchor = F.normalize(embedding.mean(dim=1), p=2, dim=-1)

            try:
                loss = contrastive_loss(
                    anchor, positive, negatives, temperature=args.temperature
                )
            except ValueError as e:
                print(f"Error in loss calculation: {e}")
                continue

            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question-file", type=str, default="./C3L/question.jsonl")
    parser.add_argument("--i2c-path", type=str, default="./C3L/I2C.pt")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--load-4bit", type=bool, default=False)
    args = parser.parse_args()

    contrastive_learning_train(args)
