
import torch 
def text_encode(tokenizer, prompt, text_encoder, tokenizer_max_length=None):
    with torch.no_grad():
        text_inputs = tokenize_prompt(tokenizer, prompt, tokenizer_max_length=tokenizer_max_length)
        prompt_embeds = encode_prompt(
            text_encoder,
            text_inputs.input_ids,
            text_inputs.attention_mask,
            text_encoder_use_attention_mask=True,
        )

    return prompt_embeds

def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs

def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds