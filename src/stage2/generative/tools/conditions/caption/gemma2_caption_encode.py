import html
import re
import urllib.parse as ul

import ftfy
import torch
from bs4 import BeautifulSoup
from diffusers.utils.import_utils import (
    BACKENDS_MAPPING,
    is_bs4_available,
    is_ftfy_available,
)
from loguru import logger

if is_bs4_available():
    from bs4 import BeautifulSoup

if is_ftfy_available():
    import ftfy


bad_punct_regex = re.compile(
    r"["
    + "#®•©™&@·º½¾¿¡§~"
    + r"\)"
    + r"\("
    + r"\]"
    + r"\["
    + r"\}"
    + r"\{"
    + r"\|"
    + "\\"
    + r"\/"
    + r"\*"
    + r"]{1,}"
)


def _clean_caption(caption):
    caption = str(caption)
    caption = ul.unquote_plus(caption)
    caption = caption.strip().lower()
    caption = re.sub("<person>", "person", caption)
    # urls:
    caption = re.sub(
        r"\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
        "",
        caption,
    )  # regex for urls
    caption = re.sub(
        r"\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
        "",
        caption,
    )  # regex for urls
    # html:
    caption = BeautifulSoup(caption, features="html.parser").text

    # @<nickname>
    caption = re.sub(r"@[\w\d]+\b", "", caption)

    # 31C0—31EF CJK Strokes
    # 31F0—31FF Katakana Phonetic Extensions
    # 3200—32FF Enclosed CJK Letters and Months
    # 3300—33FF CJK Compatibility
    # 3400—4DBF CJK Unified Ideographs Extension A
    # 4DC0—4DFF Yijing Hexagram Symbols
    # 4E00—9FFF CJK Unified Ideographs
    caption = re.sub(r"[\u31c0-\u31ef]+", "", caption)
    caption = re.sub(r"[\u31f0-\u31ff]+", "", caption)
    caption = re.sub(r"[\u3200-\u32ff]+", "", caption)
    caption = re.sub(r"[\u3300-\u33ff]+", "", caption)
    caption = re.sub(r"[\u3400-\u4dbf]+", "", caption)
    caption = re.sub(r"[\u4dc0-\u4dff]+", "", caption)
    caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)
    #######################################################

    # все виды тире / all types of dash --> "-"
    caption = re.sub(
        r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",  # noqa
        "-",
        caption,
    )

    # кавычки к одному стандарту
    caption = re.sub(r"[`´«»“”¨]", '"', caption)
    caption = re.sub(r"[‘’]", "'", caption)

    # &quot;
    caption = re.sub(r"&quot;?", "", caption)
    # &amp
    caption = re.sub(r"&amp", "", caption)

    # ip adresses:
    caption = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", caption)

    # article ids:
    caption = re.sub(r"\d:\d\d\s+$", "", caption)

    # \n
    caption = re.sub(r"\\n", " ", caption)

    # "#123"
    caption = re.sub(r"#\d{1,3}\b", "", caption)
    # "#12345.."
    caption = re.sub(r"#\d{5,}\b", "", caption)
    # "123456.."
    caption = re.sub(r"\b\d{6,}\b", "", caption)
    # filenames:
    caption = re.sub(r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", caption)

    #
    caption = re.sub(r"[\"\']{2,}", r'"', caption)  # """AUSVERKAUFT"""
    caption = re.sub(r"[\.]{2,}", r" ", caption)  # """AUSVERKAUFT"""

    caption = re.sub(bad_punct_regex, r" ", caption)  # ***AUSVERKAUFT***, #AUSVERKAUFT
    caption = re.sub(r"\s+\.\s+", r" ", caption)  # " . "

    # this-is-my-cute-cat / this_is_my_cute_cat
    regex2 = re.compile(r"(?:\-|\_)")
    if len(re.findall(regex2, caption)) > 3:
        caption = re.sub(regex2, " ", caption)

    caption = ftfy.fix_text(caption)
    caption = html.unescape(html.unescape(caption))

    caption = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", caption)  # jc6640
    caption = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", caption)  # jc6640vc
    caption = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", caption)  # 6640vc231

    caption = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", caption)
    caption = re.sub(r"(free\s)?download(\sfree)?", "", caption)
    caption = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", caption)
    caption = re.sub(
        r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", caption
    )
    caption = re.sub(r"\bpage\s+\d+\b", "", caption)

    caption = re.sub(
        r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", caption
    )  # j2d1a2a...

    caption = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", caption)

    caption = re.sub(r"\b\s+\:\s+", r": ", caption)
    caption = re.sub(r"(\D[,\./])\b", r"\1 ", caption)
    caption = re.sub(r"\s+", " ", caption)

    caption.strip()

    caption = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", caption)
    caption = re.sub(r"^[\'\_,\-\:;]", r"", caption)
    caption = re.sub(r"[\'\_,\-\:\-\+]$", r"", caption)
    caption = re.sub(r"^\.\S+$", "", caption)

    return caption.strip()


def _text_preprocessing(text: str, clean_caption=False):
    if clean_caption and not is_bs4_available():
        logger.warning(
            BACKENDS_MAPPING["bs4"][-1].format("Setting `clean_caption=True`")
        )
        logger.warning("Setting `clean_caption` to False...")
        clean_caption = False

    if clean_caption and not is_ftfy_available():
        logger.warning(
            BACKENDS_MAPPING["ftfy"][-1].format("Setting `clean_caption=True`")
        )
        logger.warning("Setting `clean_caption` to False...")
        clean_caption = False

    if not isinstance(text, (tuple, list)):
        text = [text]

    def process(text: str):
        if clean_caption:
            text = _clean_caption(text)
            text = _clean_caption(text)
        else:
            text = text.lower().strip()
        return text

    return [process(t) for t in text]


# * --- Gemma2 caption encode --- #

__gemma2_ckpt = "/home/office-401/.cache/huggingface/hub/models--Efficient-Large-Model--SANA1.5_1.6B_1024px_diffusers/snapshots/caa51e5ea874be07d3a9c7c2d0fd800570b18440/text_encoder"
__tokenizer_path = "/home/office-401/.cache/huggingface/hub/models--Efficient-Large-Model--SANA1.5_1.6B_1024px_diffusers/snapshots/caa51e5ea874be07d3a9c7c2d0fd800570b18440/tokenizer"
__auto_model_path = "/home/office-401/.cache/huggingface/hub/models--Efficient-Large-Model--SANA1.5_1.6B_1024px_diffusers/snapshots/caa51e5ea874be07d3a9c7c2d0fd800570b18440/text_encoder"

ckpt_path = __gemma2_ckpt
tokenizer_path = __tokenizer_path
auto_model_path = __auto_model_path

from transformers import AutoModelForCausalLM
from transformers.models import AutoTokenizer, Gemma2Model, T5EncoderModel, T5Tokenizer


def get_tokenizer_and_text_encoder(
    tokenizer_path=None, model_path=None, name="T5", device="cuda"
):
    text_encoder_dict = {
        "T5": "DeepFloyd/t5-v1_1-xxl",
        "T5-small": "google/t5-v1_1-small",
        "T5-base": "google/t5-v1_1-base",
        "T5-large": "google/t5-v1_1-large",
        "T5-xl": "google/t5-v1_1-xl",
        "T5-xxl": "google/t5-v1_1-xxl",
        "gemma-2b": "google/gemma-2b",
        "gemma-2b-it": "google/gemma-2b-it",
        "gemma-2-2b": "google/gemma-2-2b",
        "gemma-2-2b-it": "Efficient-Large-Model/gemma-2-2b-it",
        "gemma-2-9b": "google/gemma-2-9b",
        "gemma-2-9b-it": "google/gemma-2-9b-it",
        "Qwen2-0.5B-Instruct": "Qwen/Qwen2-0.5B-Instruct",
        "Qwen2-1.5B-Instruct": "Qwen/Qwen2-1.5B-Instruct",
    }
    assert name in list(
        text_encoder_dict.keys()
    ), f"not support this text encoder: {name}"

    tokenizer_name = encoder_name = text_encoder_dict[name]
    if tokenizer_path is not None and model_path is not None:
        tokenizer_name = tokenizer_path
        encoder_name = model_path

    if "T5" in name:
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        text_encoder = T5EncoderModel.from_pretrained(
            encoder_name, torch_dtype=torch.float16
        ).to(device)
    elif "gemma" in name or "Qwen" in name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.padding_side = "right"
        text_encoder = (
            AutoModelForCausalLM.from_pretrained(
                encoder_name, torch_dtype=torch.bfloat16
            )
            .get_decoder()
            .to(device)
        )
    else:
        print("error load text encoder")
        exit()

    return tokenizer, text_encoder


def gemma2_caption_encode(
    model_path=ckpt_path,
    tokenizer_path=tokenizer_path,
    auto_model_path=auto_model_path,
    model_name="gemma-2-2b-it",
    load_from_auto_model=True,
    max_size: int = 300,
    device="cuda",
    return_truncated=True,
):
    if (
        tokenizer_path is not None
        and model_path is not None
        and not load_from_auto_model
    ):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.padding_side = "right"
        text_encoder = Gemma2Model.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        ).to(device)
    else:
        tokenizer, text_encoder = get_tokenizer_and_text_encoder(
            name=model_name,
            tokenizer_path=tokenizer_path,
            model_path=auto_model_path,
            device=device,
        )

    select_index = [0] + list(range(-max_size + 1, 0))

    def encode(prompt: str):
        prompt = _text_preprocessing(prompt, clean_caption=False)  # type: ignore
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_size,
            truncation=True,
            return_tensors="pt",
            # add_special_tokens=True,  # no <eos>
        )
        text_input_ids: torch.Tensor = text_inputs.input_ids.to(device)
        # valid_length = text_input_ids.ne(tokenizer.pad_token_id).sum(dim=1)

        prompt_attention_mask = text_inputs.attention_mask
        prompt_attention_mask = prompt_attention_mask.to(device)
        prompt_embeds = text_encoder(
            text_input_ids, attention_mask=prompt_attention_mask
        )

        # keep the last N tokens of the embedding
        prompt_embeds = prompt_embeds[0][:, select_index]
        prompt_attention_mask = prompt_attention_mask[0, select_index]
        valid_length = prompt_attention_mask.sum().item()
        logger.debug(f"{valid_length=}")

        if return_truncated:
            truncated_embeds = prompt_embeds[:, :valid_length]
            # truncated_mask = prompt_attention_mask[:valid_length]
            return truncated_embeds, prompt_attention_mask, valid_length

        return prompt_embeds, prompt_attention_mask, valid_length

    return encode


def __test():
    encode = gemma2_caption_encode(
        model_path=ckpt_path,
        tokenizer_path=tokenizer_path,
        max_size=300,
        device="cuda:1",
    )

    text = "A beautiful sunset over the mountains"
    prompt_embeds, prompt_attention_mask, valid_length = encode(text)

    print(prompt_embeds.shape)
    print(prompt_attention_mask.shape)
    print(valid_length)


if __name__ == "__main__":
    __test()
