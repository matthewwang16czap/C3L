import argparse
import torch
import os
import json
from tqdm import tqdm
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
from PIL import Image
import re
import numpy as np
from pathlib import Path


def get_chunk(lst, n, k):
    return np.array_split(lst, n)[k]


def add_image_token(qs, mm_use_im_start_end):
    # from model.config
    if mm_use_im_start_end:
        return (
            DEFAULT_IM_START_TOKEN
            + DEFAULT_IMAGE_TOKEN
            + DEFAULT_IM_END_TOKEN
            + "\n"
            + qs
        )
    else:
        return DEFAULT_IMAGE_TOKEN + "\n" + qs


def tokenize_input(qs, has_image, tokenizer, mm_use_im_start_end, conv):
    if has_image:
        qs = add_image_token(qs, mm_use_im_start_end)
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    )
    return input_ids


def load_pretrained_model_from_path(
    model_path, model_base, load_4bit, use_flash_attn, torch_init=False
):
    if not torch_init:
        disable_torch_init()
    model_name = get_model_name_from_path(os.path.expanduser(model_path))
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        model_base,
        model_name,
        load_4bit=load_4bit,
        use_flash_attn=use_flash_attn,
        offload_folder="./offload",
    )
    return tokenizer, model, image_processor, context_len


def get_image_tensors(dataset_path, image_processor, batch):
    image_tensors = []
    for data_sample in batch:
        image_path = os.path.join(
            Path(dataset_path).expanduser(),
            data_sample["image"],
        )
        try:
            image = Image.open(image_path)
            image_tensor = image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ][0]
            image_tensors.append(image_tensor)
        except Exception as e:
            print(f"Error processing image: {e}")
    return image_tensors


instruction_prompts = [
    "Based on the given image, generate 5 in-depth reasoning questions and then answer them.",
    "Given the image, generate 5 in-depth reasoning questions and answers.",
    "Taking the image into account, generate 5 reasoning questions along with their answers.",
    "Can you come up with 5 reasoning questions based on the image and then provide the answers?",
    "After looking at the image, devise 5 reasoning questions and provide answers to them.",
    "Contemplate the image and create 5 reasoning questions with the answers provided.",
    "Analyze the image and provide 5 reasoning questions as well as their answers.",
    "Compose 5 reasoning questions using the image with their answers.",
    "Evaluate the image and create 5 comprehensive reasoning questions and their answers.",
    "Analyze the image and craft 5 effective reasoning questions and responses.",
    "Generate 5 questions based on the content of the given image and then answer them.",
    "Given the image, generate 5 questions along with their answers.",
    "From the image provided, craft 5 questions and answer them.",
    "Come up with 5 questions related to the content of the image and provide the answers.",
    "Brainstorm 5 queries associated with the image and provide the responses.",
    "Construct 5 questions based on the information presented in the image and answer them.",
    "Ask yourself 5 questions about the content of the image and respond to them.",
    "Establish 5 queries related to the content of the image and give the answers.",
    "Ask 5 questions derived from the image and then answer them.",
    "Create 5 questions about the image and answer them.",
    "Generate 5 questions to describe the image content in detail and then answer them.",
    "Considering the picture, come up with 5 questions to describe the image content in detail along with the answers.",
    "Describe the image content with 5 questions and give the responses.",
    "Come up with 5 creative questions to express the image content and then provide the answers.",
    "Draft 5 queries to address the image content and give the replies.",
    "Create 5 questions to reveal the image content and give the resolutions.",
    "Given the photo, state 5 questions that reveal the details of the image and then answer them.",
    "Ask 5 questions about what is depicted in the image and then answer them.",
    "Make up 5 queries to explain the photo in more detail and answer them.",
    "Compose 5 questions describing the subject of the image, followed by the answers.",
    "Generate 5 questions based on the content of the given image and then briefly answer them.",
    "Given the image, generate 5 questions along with the short answers.",
    "From the image provided, craft 5 questions and briefly answer them.",
    "Come up with 5 questions related to the content of the image and provide the brief answers.",
    "Brainstorm 5 queries associated with the image and provide the brief responses.",
    "Construct 5 questions based on the information presented in the image and briefly answer them.",
    "Ask yourself 5 questions about the content of the image and briefly respond to them.",
    "Establish 5 queries related to the content of the image and give the short answers.",
    "Ask 5 questions derived from the image and then briefly answer them.",
    "Create 5 questions about the image and briefly answer them.",
    "Generate 5 True or False questions based on the image and answer them with either yes or no.",
    "Construct 5 simple questions about the image and answer them with one word or phrase.",
]
