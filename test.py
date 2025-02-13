import argparse
import copy

import torch
import torch.nn.functional as F
import os
import json
from tqdm import tqdm
import shortuuid
from typing import Dict, Optional, Sequence, List
import transformers
import random
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    IGNORE_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
from llava import conversation as conversation_lib

from PIL import Image
import math
import torch.nn as nn

# Load the .pt file
file_path = "./C3L/I2C.pt"  # Replace with your file path
tensor = torch.load(file_path)


# Print the tensor
print(tensor)
