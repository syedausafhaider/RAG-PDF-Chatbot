from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class LocalLLM:
    def __init__(self, model_name: str, device: str = None):
        """
        Initialize and load the local LLM and tokenizer.

        Args:
            model_name: Hugging Face model name or local path.
            device: Device to load the model on ('cpu' or 'cuda'). Auto-detected if None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        # Ensure model is in eval mode
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
    ) -> str:
        """
        Generate text from the prompt using the loaded LLM with truncation and safety.

        Args:
            prompt: The input prompt string.
            max_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.
            repetition_penalty: Repetition penalty to reduce verbatim repeats.

        Returns:
            Generated text as a string.
        """
        # Tokenize input with truncation to fit model max length
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).to(self.device)

        # Model generate call with safe params
        output_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0.0,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated_text = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        )

        # Remove prompt part from output to keep only new tokens
        if generated_text.startswith(prompt):
            return generated_text[len(prompt) :].strip()
        else:
            # fallback to full generation if mismatch
            return generated_text.strip()


def load_llm_model(model_name: str = None) -> LocalLLM:
    """
    Helper function to load the LLM model with default or specified name.

    Args:
        model_name: Optional model name or path; defaults to config if None.

    Returns:
        Instance of LocalLLM.
    """
    from config import LLM_MODEL_NAME

    return LocalLLM(model_name or LLM_MODEL_NAME)
