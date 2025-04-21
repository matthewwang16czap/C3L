import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, distributed
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
from peft import LoraConfig, get_peft_model  # Added for LoRA


class ContrastiveDataset(Dataset):
    def __init__(self, args, tokenizer, image_processor):  # Removed model from init
        self.args = args
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
        image_path = os.path.join(
            self.args.dataset_path,
            self.args.dataset_prefix + data_sample["image"],
        )

        # Return raw data instead of precomputed embeddings
        return {
            "description": description,
            "conversations": conversations,
            "image_path": image_path,
            "positive_idx": self.I2C_tensor[idx].argmax(),
        }


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


def process_batch(model_engine, tokenizer, args, descriptions, images, convs):
    """Helper function to process batch through model"""
    input_ids_list = []
    images_tensor = []

    # Process images
    for image in images:
        try:
            pil_image = Image.open(image).convert("RGB")
            image_tensor = model_engine.module.image_processor(
                pil_image, return_tensors="pt"
            )["pixel_values"][0]
            images_tensor.append(image_tensor)
        except:
            images_tensor.append(torch.zeros(3, 224, 224))

    # Process text
    for desc in descriptions:
        conv = conv_templates[args.conv_mode].copy()
        if model_engine.module.config.mm_use_im_start_end:
            qs = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n{desc}"
        else:
            qs = f"{DEFAULT_IMAGE_TOKEN}\n{desc}"

        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        input_ids = tokenizer_image_token(
            conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        input_ids_list.append(input_ids)

    # Batch processing
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True).to(
        args.device
    )
    images_tensor = torch.stack(images_tensor).to(args.device).half()

    with torch.cuda.amp.autocast(enabled=args.fp16):
        outputs = model_engine(
            input_ids=input_ids, images=images_tensor, output_hidden_states=True
        )

    return outputs.hidden_states[-1]


def contrastive_learning_train(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    # Load model and apply LoRA first
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, args.load_4bit
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],  # Update based on your model architecture
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=args.deepspeed_config,
        args=args,
    )

    # Dataset and Dataloader
    dataset = ContrastiveDataset(args, tokenizer, image_processor)
    sampler = (
        distributed.DistributedSampler(dataset) if model_engine.world_size > 1 else None
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=4,
    )

    for epoch in range(args.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        for batch_idx, batch in enumerate(dataloader):
            model_engine.train()
            positive_indices = batch["positive_idx"]

            # Process anchor
            anchor_emb = process_batch(
                model_engine,
                tokenizer,
                args,
                batch["description"],
                batch["image_path"],
                [conv_templates[args.conv_mode].copy()],
            )

            # Process positive/negatives
            all_qa = [c for conv in batch["conversations"] for c in conv]
            all_emb = process_batch(
                model_engine,
                tokenizer,
                args,
                all_qa,
                [None] * len(all_qa),
                [conv_templates[args.conv_mode].copy()] * len(all_qa),
            )

            # Normalize embeddings
            anchor_emb = F.normalize(anchor_emb.mean(dim=1), p=2, dim=-1)
            all_emb = F.normalize(all_emb.mean(dim=1), p=2, dim=-1)

            # Compute loss
            losses = []
            for i in range(anchor_emb.size(0)):
                pos_idx = positive_indices[i]
                positive = all_emb[pos_idx].unsqueeze(0)
                negatives = torch.cat([all_emb[:pos_idx], all_emb[pos_idx + 1 :]])

                loss = contrastive_loss(
                    anchor_emb[i].unsqueeze(0).unsqueeze(0),
                    positive.unsqueeze(0),
                    negatives.unsqueeze(0),
                    args.temperature,
                )
                losses.append(loss)

            total_loss = torch.stack(losses).mean()

            # DeepSpeed backward pass
            model_engine.backward(total_loss)
            model_engine.step()

            # Logging
            if model_engine.local_rank == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {total_loss.item()}")

            # Save checkpoint
            if (
                args.save_steps
                and (batch_idx + 1) % args.save_steps == 0
                and model_engine.local_rank == 0
            ):
                model_engine.save_checkpoint(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ... (keep original arguments) ...
    # Add LoRA arguments
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--mm_projector_lr", type=float, default=2e-5)

    # Add DeepSpeed config
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    contrastive_learning_train(args)
