import torch
import torch.nn.functional as F
import argparse
import torch
import torch.nn.functional as F
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


def contrastive_loss(anchor, positive, negatives, temperature=0.2):
    """
    Compute the contrastive loss as per the given equation.

    Arguments:
    - anchor: Tensor of shape (batch_size, dim), representation of anchor data y.
    - positive: Tensor of shape (batch_size, dim), representation of positive pseudo-label h_s.
    - negatives: Tensor of shape (batch_size, num_negatives, dim), representations of negative pseudo-labels.
    - temperature: Scalar value for temperature scaling.

    Returns:
    - loss: Contrastive loss (scalar).
    """

    # Compute cosine similarity between anchor and positive
    pos_sim = F.cosine_similarity(anchor, positive, dim=-1)  # (batch_size,)

    # Compute cosine similarity between anchor and all negatives
    neg_sim = F.cosine_similarity(
        anchor.unsqueeze(1), negatives, dim=-1
    )  # (batch_size, num_negatives)

    # Apply temperature scaling
    pos_sim = pos_sim / temperature  # (batch_size,)
    neg_sim = neg_sim / temperature  # (batch_size, num_negatives)

    # Compute the denominator (sum over positive and negative samples)
    denom = torch.cat(
        [pos_sim.unsqueeze(1), neg_sim], dim=1
    )  # (batch_size, 1 + num_negatives)
    denom = torch.logsumexp(denom, dim=1)  # LogSumExp trick for numerical stability

    # Compute the final loss
    loss = -pos_sim + denom  # (batch_size,)
    return loss.mean()  # Scalar loss


def contrastive_learning_train(args, device="cuda"):
    # Model
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
    model.get_vision_tower().eval()  # Set to eval mode (optional)
    model.train()  # Set to training mode

    # I2C
    I2C_tensor = torch.load(args.i2c_path)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Initial conversation
    conv = conv_templates[args.conv_mode].copy()

    with open(os.path.expanduser(args.question_file), "r") as f:
        for idx, line in enumerate(f):
            data_sample = json.loads(line)
            optimizer.zero_grad()

            # Get anchor embedding
            input_ids = tokenizer(
                data_sample["description"], return_tensors="pt"
            ).input_ids.to(device)
            embedding = model.model.embed_tokens(input_ids)
            anchor = F.normalize(embedding.mean(dim=1), p=2, dim=-1)
            # outputs = model(input_ids)
            # embedding = outputs.hidden_states[-1].mean(
            #     dim=1
            # )
            # anchor = F.normalize(embedding, p=2, dim=-1)

            negatives = []
            for i in range(0, len(data_sample["conversations"]), 2):
                question = data_sample["conversations"][i]["value"]
                answer = data_sample["conversations"][i + 1]["value"]
                qa = f"{conv.roles[0]}: {question} {conv.roles[1]}: {answer}"
                input_ids = tokenizer(qa, return_tensors="pt").input_ids.to(device)
                embedding = model.model.embed_tokens(input_ids)
                latent_space = F.normalize(embedding.mean(dim=1), p=2, dim=-1)
                # outputs = model(input_ids)
                # embedding = outputs.hidden_states[-1].mean(
                #     dim=1
                # )
                # latent_space = F.normalize(
                #     embedding, p=2, dim=-1
                # )

                negatives.append(latent_space)

            if len(negatives) == 0:
                print(f"Skipping index {idx} due to no negatives")
                continue  # Prevent crash

            # Pick positive sample
            positive_idx = I2C_tensor[idx].argmax()
            if positive_idx >= len(negatives):
                print(f"Skipping index {idx} due to invalid positive index")
                continue

            positive = negatives[positive_idx]
            negatives.pop(positive_idx)

            # Ensure negatives tensor is valid
            if len(negatives) == 0:
                print(f"Skipping index {idx} due to no remaining negatives after pop")
                continue

            negatives = torch.stack(negatives, dim=1)

            # Compute contrastive loss
            loss = contrastive_loss(
                anchor, positive, negatives, temperature=args.temperature
            )

            # Check for NaNs
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN loss detected at index {idx}, skipping update")
                continue

            # Backward pass
            loss.backward()
            optimizer.step()

            print(f"Epoch {idx}, Loss: {loss.item()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="./C3L/question.jsonl")
    parser.add_argument("--dataset-path", type=str, default="./dataset/data")
    parser.add_argument("--dataset-prefix", type=str, default="COCO_train2014_")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--load-4bit", type=bool, default=True)
    parser.add_argument("--i2c-path", type=str, default="./C3L/I2C.pt")
    args = parser.parse_args()

    contrastive_learning_train(args)
