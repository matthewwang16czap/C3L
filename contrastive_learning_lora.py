import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import json
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

import deepspeed
from PIL import Image


class ContrastiveDataset(Dataset):
    def __init__(
        self,
        args,
        model,
        tokenizer,
        image_processor,
    ):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.data = []
        with open(args.question_file, "r") as f:
            for line in f:
                self.data.append(json.loads(line))
        self.I2C_tensor = torch.load(args.i2c_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_sample = self.data[idx]
        description = data_sample["description"]
        conversations = data_sample["conversations"]

        # Load image and preprocess
        image_path = os.path.join(
            self.args.dataset_path,
            self.args.dataset_prefix + data_sample["image"],
        )
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ][0]
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return dummy tensor
            image_tensor = torch.zeros((3, 224, 224))

        # Prepare Q&A text
        qa_list = []
        for i in range(0, len(conversations), 2):
            question = conversations[i]["value"]
            answer = conversations[i + 1]["value"]
            qa = f"question: {question}, answer: {answer}"
            qa_list.append(qa)

        # Get all embeddings
        all_embeddings = model_forward(
            args,
            qa_list,
            None,
            self.tokenizer,
            self.model,
            [conv_templates[self.args.conv_mode].copy() for _ in qa_list],
            128,
        ).hidden_states[-1]

        # Separate positive and negative
        positive_idx = self.I2C_tensor[idx].argmax()

        positive_embedding = all_embeddings[positive_idx].unsqueeze(0)

        # Mask out the positive to get negatives
        negative_embeddings = torch.cat(
            [all_embeddings[:positive_idx], all_embeddings[positive_idx + 1 :]], dim=0
        )

        # Get Anchor
        anchor_embedding = model_forward(
            args,
            [description],
            [image_tensor],
            self.tokenizer,
            self.model,
            [conv_templates[self.args.conv_mode].copy()],
            128,
        ).hidden_states[-1]

        # Get latent space
        anchor_embedding = F.normalize(
            F.relu(anchor_embedding).mean(dim=1), p=2, dim=-1
        )
        positive_embedding = F.normalize(
            F.relu(positive_embedding).mean(dim=1), p=2, dim=-1
        )
        negative_embeddings = F.normalize(
            F.relu(negative_embeddings).mean(dim=1), p=2, dim=-1
        )

        return anchor_embedding, positive_embedding, negative_embeddings


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


def model_forward(
    args, questions, image_tensors, tokenizer, model, convs, max_new_tokens
):
    batch_size = len(questions)
    input_ids_list = []
    for i in range(batch_size):
        qs = questions[i]
        if image_tensors:
            if model.config.mm_use_im_start_end:
                qs = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + qs
                )
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        convs[i].append_message(convs[i].roles[0], qs)
        convs[i].append_message(convs[i].roles[1], None)
        prompt = convs[i].get_prompt()
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).to(args.device)
        input_ids_list.append(input_ids)

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True)

    output_ids = model(
        input_ids.to(args.device),
        images=(
            torch.stack(image_tensors).half().to(args.device) if image_tensors else None
        ),
        output_hidden_states=True,
    )

    return output_ids


def contrastive_learning_train(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        load_4bit=args.load_4bit,
    )

    model.train()

    # Load dataset
    dataset = ContrastiveDataset(
        args,
        model,
        tokenizer,
        image_processor,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.per_device_train_batch_size, shuffle=True
    )

    # Initialize DeepSpeed
    if args.use_deepspeed:
        # Load the DeepSpeed config from the provided path
        with open(args.deepspeed, "r") as f:
            deepspeed_config = json.load(f)

        # Override config with any CLI arguments if necessary
        deepspeed_config["train_micro_batch_size_per_gpu"] = (
            args.per_device_train_batch_size
        )
        deepspeed_config["gradient_accumulation_steps"] = (
            args.gradient_accumulation_steps
        )

        # Initialize DeepSpeed with the model and config
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model, model_parameters=model.parameters(), config=deepspeed_config
        )
    else:
        # If not using DeepSpeed, just use standard PyTorch optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        for batch_idx, (
            anchor_embedding,
            positive_embedding,
            negative_embeddings,
        ) in enumerate(dataloader):
            model_engine.zero_grad()  # Zero out gradients

            try:
                # Compute contrastive loss
                loss = contrastive_loss(
                    anchor_embedding,
                    positive_embedding,
                    negative_embeddings,
                    temperature=args.temperature,
                )
            except ValueError as e:
                if model_engine.global_rank == 0:
                    print(f"Error in loss calculation: {e}")
                continue

            # Backpropagate loss and update model
            model_engine.backward(loss)

            # Step optimizer, handle gradient accumulation
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                model_engine.step()
            else:
                model_engine.step(
                    no_opt=True
                )  # Skip optimizer step if not accumulated yet

            if model_engine.global_rank == 0:
                # Print loss every batch
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.detach().item()}")

            # Optionally save checkpoints
            if model_engine.global_rank == 0 and batch_idx % 1000 == 0:
                model_engine.save_checkpoint(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Core model and training args
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/matthew/models/llava-v1.5-7b",
    )
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question-file", type=str, default="./C3L/question.jsonl")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/home/matthew/fiftyone/coco-2014/train/data",
    )
    parser.add_argument("--dataset-prefix", type=str, default="COCO_train2014_")
    parser.add_argument("--i2c-path", type=str, default="./C3L/I2C.pt")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--load-4bit", type=bool, default=False)
    parser.add_argument("--device", type=str, default="cuda")

    # DeepSpeed configuration args
    parser.add_argument("--use_deepspeed", action="store_true", help="Enable DeepSpeed")
    parser.add_argument(
        "--deepspeed",
        type=str,
        default="./C3L/deepspeed_config.json",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=1,
        help="Batch size per device",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints",
        help="Directory to save model checkpoints",
    )
    args = parser.parse_args()

    contrastive_learning_train(args)
