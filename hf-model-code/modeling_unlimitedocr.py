from .modeling_deepseekv2 import (
    DeepseekV2Model, DeepseekV2ForCausalLM, _fused_chunked_lm_ce,
    _td, _td_now, _td_add, _td_report,
)
from .modeling_deepseekv2 import _INFER as _DASH_INFER, configure_dash_inference
from .configuration_deepseek_v2 import DeepseekV2Config
# transformers' dynamic-module loader only copies DIRECT relative imports
# of the entry module to the HF cache for local model dirs (no recursion). Force
# sparse_gather_triton (used lazily inside modeling_deepseekv2) to be copied by
# importing it here at the entry point. The regex in transformers only matches
# ``from .xxx import yyy`` (not ``from . import xxx``), so keep this exact form.
try:
    from .sparse_gather_triton import SPARSE_BLOCK_SIZE_SENTINEL as _sgt_sentinel  # noqa: F401
except Exception:
    pass
# same reason — packing_patch is used inside modeling_deepseekv2
# (2nd-level import), so force it to be copied by importing it here at the entry.
try:
    from .packing_patch import is_packed as _pk_copy_hint  # noqa: F401
except Exception:
    pass
# Same reason for the dash overlay: _sparse_base_impl (the base implementation),
# dash_attention (entmax router) and sparse_gather_e2e_triton (fused kernel with the
# per-block bias) are 2nd-level relative imports as well, so they are only copied
# into the HF cache when the entry module imports them directly.
try:
    from ._sparse_base_impl import DeepseekV2Model as _dash_base_copy_hint  # noqa: F401
except Exception:
    pass
try:
    from .dash_attention import DashAttention as _dash_router_copy_hint  # noqa: F401
except Exception:
    pass
try:
    from .sparse_gather_e2e_triton import triton_gather_attention_e2e as _dash_e2e_copy_hint  # noqa: F401
except Exception:
    pass
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

from transformers.cache_utils import Cache
import requests
from PIL import Image, ImageOps, ImageDraw, ImageFont
from io import BytesIO
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import os
from .deepencoder import build_sam_vit_b, build_clip_l, MlpProjector
from addict import Dict
from transformers import TextStreamer
from .conversation import get_conv_template
from abc import ABC
import math
import re


@dataclass
class CausalLMOutputWithPastAndKL(CausalLMOutputWithPast):
    """ForCausalLM output with an extra `indexer_kl_loss` field so the sparse
    indexer's KL alignment loss survives DDP's pytree flatten/unflatten (which strips
    any dynamic attributes attached to a plain `CausalLMOutputWithPast`).
    Read by `swift/trainers/seq2seq_trainer.py:compute_loss`.
    """
    indexer_kl_loss: Optional[torch.FloatTensor] = None
    # raw fused-CE token-accuracy counts (correct / valid). Kept as separate
    # counts (not a pre-averaged ratio) so the trainer can token-weight the accuracy
    # exactly like swift's native compute_acc path.
    token_acc_correct: Optional[torch.FloatTensor] = None
    token_acc_valid: Optional[torch.FloatTensor] = None

from tqdm import tqdm
import numpy as np
import time


def load_image(image_path):

    try:
        image = Image.open(image_path)
        
        corrected_image = ImageOps.exif_transpose(image)
        
        return corrected_image
        
    except Exception as e:
        print(f"error: {e}")
        try:
            return Image.open(image_path)
        except:
            return None


def re_match(text):
    ref_pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(ref_pattern, text, re.DOTALL)

    det_pattern = r'(<\|det\|>\s*([A-Za-z_][\w-]*)\s*(\[[^\]]+\])\s*<\|/det\|>)'
    for full_match, label, box in re.findall(det_pattern, text, re.DOTALL):
        matches.append((full_match, label, box))

    mathes_image = []
    mathes_other = []
    for a_match in matches:
        if a_match[1].strip() == 'image' or '<|ref|>image<|/ref|>' in a_match[0]:
            mathes_image.append(a_match[0])
        else:
            mathes_other.append(a_match[0])
    return matches, mathes_image, mathes_other


def extract_coordinates_and_label(ref_text, image_width, image_height):

    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
        if cor_list and isinstance(cor_list[0], (int, float)):
            cor_list = [cor_list]
    except Exception as e:
        print(e)
        return None

    return (label_type, cor_list)


def draw_bounding_boxes(image, refs, ouput_path, image_prefix=''):

    image_width, image_height = image.size
    
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    
    # try:
    # except IOError:
    #     try:
    #         font = ImageFont.truetype("DejaVuSans.ttf", 20) 
    #     except IOError:
    font = ImageFont.load_default()

    img_idx = 0
    
    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result
                
                color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))

                color_a = color + (20, )
                for points in points_list:
                    x1, y1, x2, y2 = points

                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)

                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)

                    if label_type == 'image':
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            cropped.save(f"{ouput_path}/images/{image_prefix}{img_idx}.jpg")
                        except Exception as e:
                            print(e)
                            pass
                        img_idx += 1
                        
                    try:
                        if label_type == 'title':
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        else:
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        text_x = x1
                        text_y = max(0, y1 - 15)
                            
                        
                        text_bbox = draw.textbbox((0, 0), label_type, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], 
                                    fill=(255, 255, 255, 30))
                        
                        draw.text((text_x, text_y), label_type, font=font, fill=color)
                    except:
                        pass
        except:
            continue
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def process_image_with_refs(image, ref_texts, output_path, image_prefix=''):

    result_image = draw_bounding_boxes(image, ref_texts, output_path, image_prefix=image_prefix)
    
    return result_image





def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
    return best_ratio


def dynamic_preprocess(image, min_num=2, max_num=16, image_size=640, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    # print(target_ratios)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # print(target_aspect_ratio)
    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images, target_aspect_ratio



def normalize_transform(mean, std):
    if mean is None and std is None:
        transform = None
    elif mean is None and std is not None:
        mean = [0.] * len(std)
        transform = transforms.Normalize(mean=mean, std=std)
    elif mean is not None and std is None:
        std = [1.] * len(mean)
        transform = transforms.Normalize(mean=mean, std=std)
    else:
        transform = transforms.Normalize(mean=mean, std=std)

    return transform



def format_messages(
        conversations: List[Dict[str, str]],
        sft_format: str = "deepseek",
        system_prompt: str = "",
):
    """
    Applies the SFT template to conversation.

    Args:
        conversations (List[Dict]): A List of messages.
        sft_format (str, optional): The format of the SFT template to use. Defaults to "deepseek".
        system_prompt (str, optional): The system prompt to use in the SFT template. Defaults to "".

    Returns:
        sft_prompt (str): The formatted text.
    """

    conv = get_conv_template(sft_format)
    conv.set_system_message(system_prompt)
    for message in conversations:
        conv.append_message(message["role"], message["content"].strip())
    sft_prompt = conv.get_prompt().strip()

    return sft_prompt


def text_encode(tokenizer, text: str, bos: bool = True, eos: bool = False):
    t = tokenizer.encode(text, add_special_tokens=False)
    bos_id = 0
    eos_id = 1
    if bos:
        t = [bos_id] + t
    if eos:
        t = t + [eos_id]

    return t

def load_pil_images(conversations: List[Dict[str, str]]) -> List[Image.Image]:
    """

    Args:
        conversations (List[Dict[str, str]]): the conversations with a list of messages. An example is :
            [
                {
                    "role": "User",
                    "content": "<image_placeholder>\nExtract all information from this image and convert them into markdown format.",
                    "images": ["./examples/table_datasets.png"]
                },
                {"role": "Assistant", "content": ""},
            ]

    Returns:
        pil_images (List[PIL.Image.Image]): the list of PIL images.

    """

    pil_images = []

    for message in conversations:
        if "images" not in message:
            continue

        for image_path in message["images"]:
            # print('----------------')
            # print(image_path)
            # print('----------------')
            # exit()
            
            # pil_img = Image.open(image_path)
            pil_img = load_image(image_path)
            pil_img = pil_img.convert("RGB")
            pil_images.append(pil_img)

    return pil_images


class BaseTransform(ABC):

    def set_rng(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        pass

    @property
    def default_shape(self):
        raise NotImplementedError


class BasicImageTransform(BaseTransform):
    def __init__(
        self, 
        mean: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        std: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        normalize: bool = True
    ):
        self.mean = mean
        self.std = std
    
        transform_pipelines = [
            transforms.ToTensor()
        ]

        normalize = normalize_transform(mean, std) if normalize else nn.Identity()
        if normalize is not None:
            transform_pipelines.append(normalize)

        self.transform = transforms.Compose(transform_pipelines)
    
    def __call__(self, x):
        x = self.transform(x)
        return x

class NoEOSTextStreamer(TextStreamer):
    def on_finalized_text(self, text: str, stream_end: bool = False):

        eos_text = self.tokenizer.decode([self.tokenizer.eos_token_id], skip_special_tokens=False)
        text = text.replace(eos_text, "\n")
        print(text, flush=True, end="")


class SlidingWindowNoRepeatNgramProcessor:
    """Block n-gram repetitions within a sliding window.
    Aligned with SGLang DeepseekOCRNoRepeatNGramLogitProcessor."""
    def __init__(self, ngram_size, window, whitelist_token_ids=None):
        self.ngram_size = ngram_size
        self.window = window
        self.whitelist = set(whitelist_token_ids) if whitelist_token_ids else set()

    def __call__(self, input_ids, scores):
        for batch_idx in range(input_ids.shape[0]):
            sequence = input_ids[batch_idx].tolist()
            if len(sequence) < self.ngram_size:
                continue
            search_start = max(0, len(sequence) - self.window)
            search_end = len(sequence) - self.ngram_size + 1
            if search_end <= search_start:
                continue
            if self.ngram_size > 1:
                current_prefix = tuple(sequence[-(self.ngram_size - 1):])
            else:
                current_prefix = tuple()
            banned = set()
            for idx in range(search_start, search_end):
                ngram = sequence[idx:idx + self.ngram_size]
                if self.ngram_size == 1 or tuple(ngram[:-1]) == current_prefix:
                    banned.add(ngram[-1])
            banned.difference_update(self.whitelist)
            for token_id in banned:
                scores[batch_idx, token_id] = float('-inf')
        return scores


class TPSTextStreamer(TextStreamer):
    """Streamer that prints TPS every `interval` tokens. Set interval=0 to disable."""
    def __init__(self, tokenizer, interval=100, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.interval = interval
        self.token_count = 0
        self.start_time = None
        self.start_token_count = 0
        self.last_report_count = 0
        self.last_report_time = None

    def put(self, value):
        import time
        if hasattr(value, 'numel'):
            self.token_count += value.numel()
        else:
            self.token_count += 1
        if self.start_time is None:
            self.start_time = time.time()
            self.last_report_time = self.start_time
            self.last_report_count = self.token_count
            self.start_token_count = self.token_count
            super().put(value)
            return
        if self.interval > 0 and self.token_count - self.last_report_count >= self.interval:
            now = time.time()
            delta_tokens = self.token_count - self.last_report_count
            delta_time = now - self.last_report_time
            recent_tps = delta_tokens / delta_time if delta_time > 0 else 0
            avg_tps = (self.token_count - self.start_token_count) / (now - self.start_time) if (now - self.start_time) > 0 else 0
            print(f"\n[TPS] tokens={self.token_count}, recent={recent_tps:.1f} t/s, avg={avg_tps:.1f} t/s", flush=True)
            self.last_report_count = self.token_count
            self.last_report_time = now
        super().put(value)

    def on_finalized_text(self, text: str, stream_end: bool = False):
        eos_text = self.tokenizer.decode([self.tokenizer.eos_token_id], skip_special_tokens=False)
        text = text.replace(eos_text, "\n")
        print(text, flush=True, end="")


@torch.no_grad()
def _uocr_dash_decode_allow(indexer, input_ids, cache):
    """Page-aware remote candidate blocks of the DECODE query (vLLM page_state.py parity).

    Prefill (q_len > 1): record, per KV block, the [min, max] image page it holds, the
    prompt page count ``pp`` (<PAGE> markers up to the last image token) and the marker
    count. Returns None (prefill builds its own per-query candidates).
    Decode (q_len == 1): the query token's page is ``marker_cum - 1 - pp`` (response pages
    restart at 0, so text page k <-> image page k); the candidate set is the blocks holding
    an image token of that page, or every block when the page is unknown / has no image
    block (training's per-query fallback). Returns [B, nblk] bool.
    """
    img_id, pg_id = indexer.image_token_id, indexer.page_token_id
    if img_id is None or pg_id is None:
        return None
    bs = int(indexer.block_size)
    ids = input_ids
    B, L = ids.shape
    if L > 1:
        img = ids == img_id
        cum = torch.cumsum((ids == pg_id).long(), dim=-1)
        run = torch.cumsum(img.long(), dim=-1)
        per_page = max(1, int(indexer.per_page_image_tokens))
        page = torch.where(cum > 0, cum - 1, (run - 1) // per_page)
        page = torch.where(img, page, torch.full_like(page, -1))
        pp = torch.where(img, cum, torch.zeros_like(cum)).cummax(dim=-1).values[:, -1]
        nblk = -(-L // bs)
        pg = F.pad(page, (0, nblk * bs - L), value=-1).view(B, nblk, bs)
        bmax = pg.amax(dim=-1)
        bmin = torch.where(pg >= 0, pg, torch.full_like(pg, 1 << 30)).amin(dim=-1)
        bmin = torch.where(bmax >= 0, bmin, torch.full_like(bmin, -1))
        cache._uocr_pg = {'bmin': bmin, 'bmax': bmax, 'pp': pp, 'mcum': cum[:, -1].clone()}
        return None
    st = getattr(cache, '_uocr_pg', None)
    if st is None:
        return None
    st['mcum'] = st['mcum'] + (ids[:, -1] == pg_id).long()
    qp = st['mcum'] - 1 - st['pp']
    qp = torch.where(qp >= 0, qp, torch.full_like(qp, -1))[:, None]
    total = int(cache.get_seq_length()) + L
    nblk = -(-total // bs)
    bmin, bmax = st['bmin'], st['bmax']
    same = (bmin >= 0) & (bmin <= qp) & (bmax >= qp) & (qp >= 0)
    open_row = (qp[:, 0] < 0) | ~same.any(dim=-1)
    allow = torch.zeros(B, nblk, dtype=torch.bool, device=ids.device)
    n0 = min(nblk, same.shape[-1])
    allow[:, :n0] = same[:, :n0]
    return allow | open_row[:, None]


def _uocr_build_inputs(tokenizer, prompt, images, crop_mode=None, base_size=1024, image_size=640,
                       min_crops=2, max_crops=16, patch_size=16, downsample_ratio=4,
                       image_token='<image>', image_token_id=128815, bos_id=0):
    """Tokens + pixel tensors exactly like vLLM's UnlimitedOCRProcessor.tokenize_with_images.

    crop_mode None == vLLM auto policy: gundam (dynamic tiles) for a single image, base
    (one letterbox-padded global view) for multi-image. No 640x640 pre-resize in base mode
    (vLLM's training-parity fix): the original page is padded straight into base_size.
    """
    n_img = len(images)
    cropping = (n_img == 1) if crop_mode is None else bool(crop_mode)
    splits = prompt.split(image_token)
    assert len(splits) == n_img + 1, f'prompt has {len(splits) - 1} {image_token} but {n_img} images'
    transform = BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)
    pad_color = tuple(int(x * 255) for x in transform.mean)
    nq = math.ceil((image_size // patch_size) / downsample_ratio)
    nqb = math.ceil((base_size // patch_size) / downsample_ratio)
    ids, seq_mask, globals_, crops, spatial = [], [], [], [], []
    for text_sep, image in zip(splits, images):
        t = tokenizer.encode(text_sep, add_special_tokens=False)
        ids += t
        seq_mask += [False] * len(t)
        tiles = []
        if image.size[0] <= image_size and image.size[1] <= image_size:
            ratio = (1, 1)
        elif cropping:
            tiles, ratio = dynamic_preprocess(image, min_num=min_crops, max_num=max_crops,
                                              image_size=image_size)
        else:
            ratio = (1, 1)
        globals_.append(transform(ImageOps.pad(image, (base_size, base_size), color=pad_color)).to(torch.bfloat16))
        w_t, h_t = int(ratio[0]), int(ratio[1])
        spatial.append([w_t, h_t])
        tok = [image_token_id] * (nqb * (nqb + 1)) + [image_token_id]
        if w_t > 1 or h_t > 1:
            crops += [transform(c).to(torch.bfloat16) for c in tiles]
            tok += ([image_token_id] * (nq * w_t + 1)) * (nq * h_t)
        ids += tok
        seq_mask += [True] * len(tok)
    t = tokenizer.encode(splits[-1], add_special_tokens=False)
    ids = [bos_id] + ids + t
    seq_mask = [False] + seq_mask + [False] * len(t)
    input_ids = torch.LongTensor(ids)
    seq_mask = torch.tensor(seq_mask, dtype=torch.bool)
    if n_img == 0:
        return input_ids, None, seq_mask, None
    images_ori = torch.stack(globals_, dim=0)
    images_crop = (torch.stack(crops, dim=0) if crops
                   else torch.zeros((1, 3, base_size, base_size), dtype=torch.bfloat16))
    return input_ids, (images_crop, images_ori), seq_mask, torch.tensor(spatial, dtype=torch.long)


def _uocr_page_count_guard(logits, gen_ids, expected_pages, page_id, eos_id, max_page_tokens, st):
    """vLLM unlimited_sparse_plugin.page_count_guard._PageCountGuard (per-step, in place)."""
    if expected_pages <= 0:
        return logits
    n = len(gen_ids)
    if n > st['scanned']:
        tail = gen_ids[st['scanned']:n]
        m = tail.count(page_id)
        if m:
            st['marker_cum'] += m
            st['page_len'] = tail[::-1].index(page_id)
        else:
            st['page_len'] += len(tail)
        st['scanned'] = n
    mc = st['marker_cum']
    if mc > expected_pages:
        keep = logits[eos_id].clone(); logits[:] = -float('inf'); logits[eos_id] = keep
    elif 0 < max_page_tokens <= st['page_len']:
        tid = page_id if mc < expected_pages else eos_id
        keep = logits[tid].clone(); logits[:] = -float('inf'); logits[tid] = keep
    elif mc < expected_pages:
        logits[eos_id] = -float('inf')
    return logits


def _uocr_no_repeat_ngram(logits, gen_ids, ngram_size, window_size, whitelist):
    """vLLM deepseek_ocr.NoRepeatNGramLogitsProcessor (on generated ids only)."""
    if ngram_size <= 0 or len(gen_ids) < ngram_size:
        return logits
    prefix = tuple(gen_ids[-(ngram_size - 1):]) if ngram_size > 1 else tuple()
    start = max(0, len(gen_ids) - window_size)
    end = len(gen_ids) - ngram_size + 1
    banned = set()
    for i in range(start, end):
        ng = tuple(gen_ids[i:i + ngram_size])
        if ng[:-1] == prefix:
            banned.add(ng[-1])
    banned -= whitelist
    if banned:
        logits[list(banned)] = -float('inf')
    return logits


class UnlimitedOCRConfig(DeepseekV2Config):
    model_type = "unlimited-ocr"

class UnlimitedOCRModel(DeepseekV2Model):
    config_class = UnlimitedOCRConfig

    def __init__(self, config: DeepseekV2Config):
        super(UnlimitedOCRModel, self).__init__(config)

        self.sam_model = build_sam_vit_b()
        self.vision_model = build_clip_l()
        # self.conv_2 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=2, stride=2)
        n_embed = 1280
        self.projector =  MlpProjector(Dict(projector_type="linear", input_dim=2048, n_embed=n_embed))
        embed_std = 1 / torch.sqrt(torch.tensor(n_embed, dtype=torch.float32))
        self.image_newline = nn.Parameter(torch.randn(n_embed) * embed_std)
        self.view_seperator = nn.Parameter(torch.randn(n_embed) * embed_std)



    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        images_seq_mask: Optional[torch.FloatTensor] = None,
        images_spatial_crop: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # hand the full token id sequence directly to each sparse attention module so the
        # indexer can derive per-page image/query ids (page-aware image bucket constraint).
        # Set on the attention module itself (not config) to avoid any config-object identity
        # ambiguity and to survive gradient-checkpointing recompute.
        if input_ids is not None and getattr(input_ids, 'ndim', 0) == 2:
            _rope2d = None
            _layers = list(self.layers)
            _first = next((getattr(l, 'self_attn', None) for l in _layers
                           if getattr(getattr(l, 'self_attn', None), 'indexer', None) is not None),
                          None)
            if (_first is not None and input_ids.shape[-1] > 1
                    and getattr(_first, 'main_rope_mode', '1d') == 'page2d'):
                _rope2d = _first.indexer._build_page_rope_pos(input_ids, input_ids.shape[-1])
            # The per-page prefill counts (pf_cnt) that seed the decode-side page-rope state
            # machine (_main_rope_coords) are written only on the indexer that built the
            # coordinates (layer 0); hand them to every layer, otherwise layers 1.. fall
            # back to 1D RoPE as soon as decoding starts.
            _meta = (getattr(_first.indexer, '_page_rope_meta', None)
                     if (_first is not None and _rope2d is not None) else None)
            # infer(): page-aware candidate blocks of the decode query (vLLM page_state parity).
            _dec_allow = None
            if _DASH_INFER["on"] and past_key_values is not None and _first is not None:
                _dec_allow = _uocr_dash_decode_allow(_first.indexer, input_ids, past_key_values)
            for _layer in _layers:
                _attn = getattr(_layer, 'self_attn', None)
                if _attn is not None and getattr(_attn, 'indexer', None) is not None:
                    _attn._sparse_source_input_ids = input_ids
                    _attn._main_rope_pos_2d = _rope2d
                    if _meta is not None:
                        _attn.indexer._page_rope_meta = _meta
                    _attn._dash_dec_allow = _dec_allow

        if inputs_embeds is None:
            # inputs_embeds = self.embed_tokens(input_ids)
            inputs_embeds = self.get_input_embeddings()(input_ids)



        sam_model = getattr(self, 'sam_model', None)
        # sam_model = self.sam_model
        vision_model = getattr(self, 'vision_model', None)



        # TIME_DIAG=1 -> attribute the per-page vision encoder cost.
        _t_vis = _td_now()
        if sam_model is not None and images is not None and (input_ids.shape[1] != 1 or self.training) and torch.sum(images[0][1]).item() != 0:

            _packed_mm = (inputs_embeds.shape[0] == 1 and len(images) > 1)
            _packed_feats = []
            for idx, image in enumerate(images):
                images_in_this_batch = []
                patches_all = image[0]
                image_ori = image[1]
                spatial = image[2] if len(image) > 2 else None

                num_pages = image_ori.shape[0]
                tile_cursor = 0
                with torch.no_grad():
                    for page_idx in range(num_pages):
                        image_ori_p = image_ori[page_idx:page_idx + 1]
                        if spatial is not None:
                            width_crop_num = int(spatial[page_idx][0])
                            height_crop_num = int(spatial[page_idx][1])
                        else:
                            width_crop_num = height_crop_num = 1
                        do_crop = (width_crop_num > 1 or height_crop_num > 1) and torch.sum(patches_all).item() != 0

                        if do_crop:
                            n_tiles = width_crop_num * height_crop_num
                            patches = patches_all[tile_cursor:tile_cursor + n_tiles]
                            tile_cursor += n_tiles
                            local_features_1 = sam_model(patches)
                            local_features_2 = vision_model(patches, local_features_1)
                            local_features = torch.cat((local_features_2[:, 1:], local_features_1.flatten(2).permute(0, 2, 1)), dim=-1)
                            local_features = self.projector(local_features)

                            global_features_1 = sam_model(image_ori_p)
                            global_features_2 = vision_model(image_ori_p, global_features_1)
                            global_features = torch.cat((global_features_2[:, 1:], global_features_1.flatten(2).permute(0, 2, 1)), dim=-1)
                            global_features = self.projector(global_features)

                            _, hw, n_dim = global_features.shape
                            h = w = int(hw ** 0.5)
                            _2, hw2, n_dim2 = local_features.shape
                            h2 = w2 = int(hw2 ** 0.5)

                            global_features = global_features.view(h, w, n_dim)
                            global_features = torch.cat(
                                [global_features, self.image_newline[None, None, :].expand(h, 1, n_dim)], dim=1
                            )
                            global_features = global_features.view(-1, n_dim)

                            local_features = local_features.view(height_crop_num, width_crop_num, h2, w2, n_dim2).permute(0, 2, 1, 3, 4).reshape(height_crop_num * h2, width_crop_num * w2, n_dim2)
                            local_features = torch.cat(
                                [local_features, self.image_newline[None, None, :].expand(height_crop_num * h2, 1, n_dim2)], dim=1
                            )
                            local_features = local_features.view(-1, n_dim2)

                            global_local_features = torch.cat([local_features, global_features, self.view_seperator[None, :]], dim=0)
                            images_in_this_batch.append(global_local_features)
                        else:
                            global_features_1 = sam_model(image_ori_p)
                            global_features_2 = vision_model(image_ori_p, global_features_1)
                            global_features = torch.cat((global_features_2[:, 1:], global_features_1.flatten(2).permute(0, 2, 1)), dim=-1)
                            global_features = self.projector(global_features)

                            _, hw, n_dim = global_features.shape
                            h = w = int(hw ** 0.5)
                            global_features = global_features.view(h, w, n_dim)
                            global_features = torch.cat(
                                [global_features, self.image_newline[None, None, :].expand(h, 1, n_dim)], dim=1
                            )
                            global_features = global_features.view(-1, n_dim)
                            global_local_features = torch.cat([global_features, self.view_seperator[None, :]], dim=0)
                            images_in_this_batch.append(global_local_features)

                if images_in_this_batch:
                    images_in_this_batch = torch.cat(images_in_this_batch, dim=0)
                    if _packed_mm:
                        _packed_feats.append(images_in_this_batch)
                    else:
                        inputs_embeds[idx].masked_scatter_(images_seq_mask[idx].unsqueeze(-1).cuda(), images_in_this_batch)

            if _packed_mm and _packed_feats:
                _packed_feats = torch.cat(_packed_feats, dim=0)
                inputs_embeds[0].masked_scatter_(images_seq_mask[0].unsqueeze(-1).cuda(), _packed_feats)

        _td_add("vision", _t_vis)

        with _td("llm_decoder"):
            return super(UnlimitedOCRModel, self).forward(
                input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
                inputs_embeds=inputs_embeds, use_cache=use_cache, position_ids = position_ids,
                output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
    

class UnlimitedOCRForCausalLM(DeepseekV2ForCausalLM):

    config_class = UnlimitedOCRConfig
    # supports_gradient_checkpointing = True

    def __init__(self, config):
        super(DeepseekV2ForCausalLM, self).__init__(config)
        self.model = UnlimitedOCRModel(config)

        self.vocab_size = config.vocab_size

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        images_seq_mask: Optional[torch.FloatTensor] = None,
        images_spatial_crop: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict



        outputs  = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            images=images,
            images_seq_mask = images_seq_mask,
            images_spatial_crop = images_spatial_crop,
            return_dict=return_dict
            
        )


        
        # print(transformer_outputs)

        hidden_states = outputs[0]

        # fused chunked linear cross-entropy. Avoids materialising the full
        # [B, seq, vocab=129280] logits (and its fp32 upcast) that OOMs on long OCR sequences.
        # Gated by FUSED_CHUNK_CE (default on when labels present); falls back to the classic
        # full-logits path so generation / return-logits callers stay unaffected.
        _use_fused_ce = (
            labels is not None
            and os.environ.get('FUSED_CHUNK_CE', '1') == '1'
        )

        logits = None
        loss = None
        token_acc_correct = None
        token_acc_valid = None
        if _use_fused_ce:
            with _td("ce"):
                loss, token_acc_correct, token_acc_valid = _fused_chunked_lm_ce(
                    hidden_states,
                    self.lm_head,
                    labels,
                    self.config.vocab_size,
                    num_items_in_batch=None,
                    chunk_size=int(os.environ.get('CE_CHUNK_SIZE', '1024')),
                    return_token_acc=True,
                )
            _td_report(f"seq={hidden_states.shape[1]}")
        else:
            logits = self.lm_head(hidden_states)
            logits = logits.float()

            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

        # propagate indexer_kl_loss to outer output; trainer is responsible for
        # weighting/adding it into the final loss and logging. Do NOT add KL here to
        # avoid double-counting when swift's compute_loss picks it up.
        _kl = getattr(outputs, 'indexer_kl_loss', None)
        # one-shot diagnostic
        if not getattr(self, '_kl_prop_debug_logged', False):
            self._kl_prop_debug_logged = True
            try:
                _inner_type = type(outputs).__name__
                _kl_repr = 'None' if _kl is None else f'tensor(shape={list(_kl.shape)}, dtype={_kl.dtype}, val={float(_kl.detach()):.6f})'
                print(f"[debug ForCausalLM] inner_outputs_type={_inner_type} "
                      f"inner_has_indexer_kl={hasattr(outputs, 'indexer_kl_loss')} "
                      f"_kl={_kl_repr} labels_not_None={labels is not None}", flush=True)
            except Exception as _e:
                print(f"[debug ForCausalLM] failed: {_e}", flush=True)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        result = CausalLMOutputWithPastAndKL(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            indexer_kl_loss=_kl,  # proper dataclass field, survives DDP pytree
            token_acc_correct=token_acc_correct,  # fused-CE correct token count
            token_acc_valid=token_acc_valid,      # fused-CE valid token count
        )
        return result


    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # Omit tokens covered by past_key_values
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.get_seq_length()
                max_cache_length = getattr(past_key_values, 'get_max_length', lambda: None)()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Ring buffer: cache size is fixed, but we've processed more tokens.
            # Always just take the last token for decode.
            if hasattr(past_key_values, '_prefill_length') and past_length > 0:
                input_ids = input_ids[:, -1:]
            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            elif attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if self.generation_config.cache_implementation == "static":
        #     # generation with static cache
        #     cache_position = kwargs.get("cache_position", None)
        #     if cache_position is None:
        #         past_length = 0
        #     else:
        #         past_length = cache_position[-1] + 1
        #     input_ids = input_ids[:, past_length:]
        #     position_ids = position_ids[:, past_length:]

        # TODO @gante we should only keep a `cache_position` in generate, and do +=1.
        # same goes for position ids. Could also help with continued generation.
        cache_position = torch.arange(past_length, past_length + position_ids.shape[-1], device=position_ids.device)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # Pass images only on prefill (cache empty or None)
        _is_prefill = (past_key_values is None or
                       (isinstance(past_key_values, Cache) and past_key_values.get_seq_length() == 0))
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None) if _is_prefill else None,
                "images_seq_mask": kwargs.get("images_seq_mask", None) if _is_prefill else None,
                "images_spatial_crop": kwargs.get("images_spatial_crop", None) if _is_prefill else None,
            }
        )
        return model_inputs
    

    def disable_torch_init(self):
        """
        Disable the redundant torch default initialization to accelerate model creation.
        """
        import torch
        setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
        setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)



    @torch.no_grad()
    def infer(self, tokenizer, prompt='', image_file='', output_path='', base_size=1024, image_size=640,
              crop_mode=None, test_compress=False, save_results=False, eval_mode=False, max_length=131072,
              max_new_tokens=None, tps_interval=0, no_repeat_ngram_size=0, ngram_window=0,
              ngram_whitelist_token_ids=None, temperature=0.0, page_count_guard=True,
              max_page_tokens=20000, dash_topm=16, max_crops=16, return_dict=False):
        """Single-/multi-page inference, aligned with the vLLM deployment
        (sparse_attn_infer-451/deploy_multimodel_dash_01.sh + request_openai*.py).

        image_file: one path / PIL image, or a list of them (one per page, in order).
        prompt: instruction text. Without ``<image>`` placeholders the trained multi-page
            layout is built: ``"<PAGE><image>" * n_pages + prompt`` (default prompt
            "Multi page parsing."). A prompt with exactly n_pages ``<image>`` is used as is;
            a single ``<image>`` with several pages is expanded to ``<PAGE><image>`` per page.
        crop_mode: None -> vLLM auto (gundam tiles for one page, base for several);
            True / False forces gundam / base.
        Generation mirrors vLLM: greedy when temperature == 0, stops at EOS or
        ``max_new_tokens`` (default ``max_length - prompt_len``), PageCountGuard on by default
        (prompt <PAGE> count == output pages; per-page cap ``max_page_tokens``), optional
        NoRepeatNGram (``no_repeat_ngram_size`` / ``ngram_window``, vLLM extra_args semantics).
        Decoder attention = DASH block-sparse in prefill and decode (vLLM DASH_DECODE_DENSE=0,
        DASH_INFER_TOPM=dash_topm, DASH_PAGE_VISIBLE_BACK=0).

        Returns the generated text (special tokens kept, EOS dropped -- like vLLM with
        skip_special_tokens=False), or a dict with token ids when ``return_dict``.
        """
        if isinstance(image_file, (list, tuple)):
            files = [f for f in image_file if f is not None and f != '']
        elif image_file is None or image_file == '':
            files = []
        else:
            files = [image_file]
        images = [f.convert('RGB') if isinstance(f, Image.Image) else Image.open(f).convert('RGB')
                  for f in files]
        n_img = len(images)
        prompt = prompt or ('Multi page parsing.' if n_img else '')
        if not prompt:
            raise ValueError('prompt is empty!')
        n_ph = prompt.count('<image>')
        if n_img and n_ph == 0:
            prompt = '<PAGE><image>' * n_img + prompt
        elif n_img > 1 and n_ph == 1:
            prompt = prompt.replace('<image>', '<PAGE><image>' * n_img)
        elif n_ph != n_img:
            raise ValueError(f'prompt has {n_ph} <image> placeholders but {n_img} images')

        input_ids, imgs, seq_mask, spatial = _uocr_build_inputs(
            tokenizer, prompt, images, crop_mode=crop_mode, base_size=base_size,
            image_size=image_size, max_crops=max_crops)
        res = self._uocr_generate(
            tokenizer, input_ids, imgs, seq_mask, spatial,
            max_new_tokens=(max_new_tokens if max_new_tokens else max(1, max_length - input_ids.numel())),
            max_length=max_length, temperature=temperature, page_count_guard=page_count_guard,
            max_page_tokens=max_page_tokens, ngram_size=no_repeat_ngram_size,
            ngram_window=(ngram_window if ngram_window > 0 else 100),
            ngram_whitelist=set(ngram_whitelist_token_ids or []), dash_topm=dash_topm,
            stream=not eval_mode, tps_interval=tps_interval)
        outputs = res['text']
        if test_compress and n_img:
            out_len = len(tokenizer.encode(outputs, add_special_tokens=False))
            n_vis = int(seq_mask.sum())
            print('=' * 50)
            print('image tokens: ', n_vis, ' output tokens: ', out_len,
                  ' compression ratio: ', round(out_len / max(1, n_vis), 2))
            print('=' * 50)
        if save_results and output_path:
            self._uocr_save_results(outputs, images, output_path)
        if return_dict:
            res['prompt'] = prompt
            return res
        return outputs

    def infer_multi(self, tokenizer, prompt='', image_files=None, output_path='', image_size=640,
                    save_results=False, max_length=131072, tps_interval=0, no_repeat_ngram_size=0,
                    ngram_window=0, temperature=0.0, **kwargs):
        """Backward-compatible multi-page entry point -> ``infer`` (base mode, no crop).
        Returns (outputs, output_tokens) like before."""
        if not image_files:
            raise ValueError('image_files must be a non-empty list for multi-image inference!')
        kwargs.setdefault('crop_mode', False)
        res = self.infer(tokenizer, prompt=prompt.replace('<image>', '') if prompt.count('<image>') == 1 else prompt,
                         image_file=list(image_files), output_path=output_path, image_size=image_size,
                         save_results=save_results, max_length=max_length, tps_interval=tps_interval,
                         no_repeat_ngram_size=no_repeat_ngram_size, ngram_window=ngram_window,
                         temperature=temperature, return_dict=True, **kwargs)
        return res['text'], len(res['token_ids'])

    @torch.no_grad()
    def _uocr_generate(self, tokenizer, input_ids, imgs, seq_mask, spatial, max_new_tokens=16384,
                       max_length=131072, temperature=0.0, page_count_guard=True, max_page_tokens=20000,
                       ngram_size=0, ngram_window=100, ngram_whitelist=None, dash_topm=16,
                       stream=False, tps_interval=0):
        """Prefill + token-by-token decode on a DynamicCache (no HF generate()).
        Only the last position goes through lm_head (full-sequence logits at 30k tokens x
        129k vocab would not fit)."""
        import time
        from transformers.cache_utils import DynamicCache
        device = self.lm_head.weight.device
        ids = input_ids.unsqueeze(0).to(device)
        prompt_ids = input_ids.tolist()
        page_id = int((self.config.sparse_attention_config or {}).get('page_token_id', 128827)) \
            if isinstance(getattr(self.config, 'sparse_attention_config', None), dict) else 128827
        eos_id = int(tokenizer.eos_token_id)
        expected_pages = prompt_ids.count(page_id) if page_count_guard else 0
        guard_st = {'marker_cum': 0, 'scanned': 0, 'page_len': 0}
        kw = {}
        if imgs is not None:
            kw = dict(images=[(imgs[0].to(device), imgs[1].to(device), spatial)],
                      images_seq_mask=seq_mask.unsqueeze(0).to(device),
                      images_spatial_crop=spatial)

        # decode-time knobs: no ring-buffer SWA (vLLM attends via DASH), vLLM-parity router
        cfg = self.config
        saved_cfg = (getattr(cfg, '_ring_window', None), getattr(cfg, 'sliding_window', None))
        cfg._ring_window = None
        cfg.sliding_window = None
        dash_state = configure_dash_inference(self.model, True, topm=dash_topm)
        streamer = TextStreamer(tokenizer, skip_prompt=False, skip_special_tokens=False) if stream else None
        gen, finish = [], 'length'
        t0 = time.time()
        t_first = None
        try:
            cache = DynamicCache()
            S = ids.shape[1]
            out = self.model(input_ids=ids, past_key_values=cache, use_cache=True,
                             position_ids=torch.arange(S, device=device)[None], return_dict=True, **kw)
            cache = out.past_key_values
            h = out.last_hidden_state[:, -1:]
            for step in range(max_new_tokens):
                logits = self.lm_head(h).float()[0, -1]
                if ngram_size > 0:
                    logits = _uocr_no_repeat_ngram(logits, gen, ngram_size, ngram_window, ngram_whitelist or set())
                if expected_pages > 0:
                    logits = _uocr_page_count_guard(logits, gen, expected_pages, page_id, eos_id,
                                                    max_page_tokens, guard_st)
                if temperature and temperature > 0:
                    nxt = int(torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1))
                else:
                    nxt = int(torch.argmax(logits))
                if t_first is None:
                    t_first = time.time()
                if nxt == eos_id:
                    finish = 'stop'
                    break
                gen.append(nxt)
                if streamer is not None:
                    streamer.put(torch.tensor([nxt]))
                if tps_interval and len(gen) % tps_interval == 0:
                    dt = time.time() - t_first
                    print(f'\n[TPS] tokens={len(gen)} avg={len(gen) / max(dt, 1e-6):.1f} t/s', flush=True)
                pos = S + len(gen) - 1
                if pos + 1 >= max_length:
                    break
                out = self.model(input_ids=torch.tensor([[nxt]], device=device), past_key_values=cache,
                                 use_cache=True, position_ids=torch.tensor([[pos]], device=device),
                                 return_dict=True)
                cache = out.past_key_values
                h = out.last_hidden_state
        finally:
            configure_dash_inference(self.model, False, state=dash_state)
            cfg._ring_window, cfg.sliding_window = saved_cfg
            if streamer is not None:
                streamer.end()
        text = tokenizer.decode(gen, skip_special_tokens=False)
        return {'text': text, 'token_ids': gen, 'prompt_token_ids': prompt_ids,
                'finish_reason': finish, 'num_pages_out': gen.count(page_id),
                'expected_pages': expected_pages, 'time': time.time() - t0,
                'prefill_time': (t_first - t0) if t_first else None}

    def _uocr_save_results(self, outputs, images, output_path):
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(f'{output_path}/images', exist_ok=True)
        pages = outputs.split('<PAGE>')[1:] if '<PAGE>' in outputs else [outputs]
        processed = []
        for page_idx, page_output in enumerate(pages):
            page_output = page_output.strip()
            if page_idx < len(images):
                matches_ref, matches_images, mathes_other = re_match(page_output)
                prefix = f'page_{page_idx}_'
                result = process_image_with_refs(images[page_idx].copy(), matches_ref, output_path,
                                                 image_prefix=prefix)
                result.save(f'{output_path}/result_with_boxes_{page_idx}.jpg')
                for idx, m in enumerate(matches_images):
                    page_output = page_output.replace(m, f'![](images/{prefix}{idx}.jpg)\n')
                for m in mathes_other:
                    page_output = page_output.replace(m, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')
            processed.append(page_output)
        with open(f'{output_path}/result.md', 'w', encoding='utf-8') as f:
            f.write('<PAGE>\n' + '\n<PAGE>\n'.join(processed))
        with open(f'{output_path}/result_raw.md', 'w', encoding='utf-8') as f:
            f.write(outputs)
