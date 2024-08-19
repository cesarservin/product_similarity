import numpy as np  # type: ignore


class TokenizerProcessorMLM:
    from transformers import PreTrainedTokenizer

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length=512, mask_probability=0.15):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_probability = mask_probability

    def __call__(self, batch):
        import torch

        encodings = self.tokenizer(
            batch["text"], truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        # Custom code for MLM masking
        labels = input_ids.clone()
        rand = torch.rand(input_ids.shape)
        mask = (rand < self.mask_probability) & (input_ids != self.tokenizer.pad_token_id)
        input_ids[mask] = self.tokenizer.mask_token_id

        return {
            "input_ids": input_ids.tolist(),  # Convert tensors to lists for compatibility with Dataset
            "attention_mask": attention_mask.tolist(),
            "labels": labels.tolist(),
        }


def get_embedding(text: str, tokenizer: dict, model: str, device: str) -> np.ndarray:
    """Get the mean embedding of a text using a transformer model

    Args:
        text (str): text to embed
        tokenizer (dict): tokenizer object
        model (str): model object
        device (str): device to use

    Returns:
        np.ndarray: mean embedding of the text
    """
    import torch

    # Tokenize and move to device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        # Forward pass to get model outputs
        outputs = model(**inputs)

        # Access last_hidden_state
        last_hidden_state = outputs.last_hidden_state
        # Compute mean of last hidden state and convert to numpy array
        mean_embedding = last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    return mean_embedding
