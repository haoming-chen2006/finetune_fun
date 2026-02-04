"""Chat Application for Fine-tuned Language Models

Interactive chat interface to talk with fine-tuned LoRA models.
Supports multiple model checkpoints (19th century, persona, etc.)

Usage:
    python app/chat.py persona        # Persona-conditioned chat
    python app/chat.py 19thcentury    # 19th century literature style
    python app/chat.py standard       # Base model (no fine-tuning)

Options:
    --8bit / --4bit    Load with quantization
    --checkpoint PATH  Custom checkpoint path
"""

import sys
sys.path.insert(0, "/home/haoming/finetune_fun")

import argparse
from pathlib import Path
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# Special tokens for persona model
PERSONA_START = "<persona>"
PERSONA_END = "</persona>"
USER1_START = "<user1>"
USER1_END = "</user1>"
USER2_START = "<user2>"
USER2_END = "</user2>"

# Default persona - set to None to prompt user, or set a string to use automatically
DEFAULT_PERSONA = None  # e.g. "I am a helpful assistant. I love coding and explaining things clearly."

PERSONA_SPECIAL_TOKENS = [PERSONA_START, PERSONA_END, USER1_START, USER1_END, USER2_START, USER2_END]


# Default model configurations
MODEL_CONFIGS = {
    "19thcentury": {
        "base_model": "Qwen/Qwen3-8B",
        "checkpoint": "runs/19thcentury/final",
        "system_prompt": "You are a knowledgeable assistant trained on 19th century British literature. Respond in a thoughtful, literary style.",
        "use_persona": False,
    },
    "persona": {
        "base_model": "Qwen/Qwen3-8B", 
        "checkpoint": "runs/persona/final",
        "system_prompt": "",
        "use_persona": True,
    },
    "standard": {
        "base_model": "Qwen/Qwen3-8B",
        "checkpoint": None,  # No fine-tuning, just base model
        "system_prompt": "You are a helpful assistant.",
        "use_persona": False,
    },
}


class ChatBot:
    """Interactive chatbot using fine-tuned LoRA model."""
    
    def __init__(
        self,
        base_model: str,
        checkpoint_path: Optional[str] = None,
        system_prompt: str = "",
        device: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        use_persona: bool = False,
    ):
        self.device = device
        self.system_prompt = system_prompt
        self.conversation_history = []
        self.use_persona = use_persona
        self.persona = None  # Set via set_persona()
        
        print(f"Loading base model: {base_model}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Add special tokens for persona mode
        if use_persona:
            self.tokenizer.add_special_tokens({"additional_special_tokens": PERSONA_SPECIAL_TOKENS})
            print(f"Added special tokens: {PERSONA_SPECIAL_TOKENS}")
        
        # Load model with quantization options
        load_kwargs = {
            "torch_dtype": "auto",
            "device_map": device,
        }
        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            load_kwargs["load_in_4bit"] = True
        
        self.model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
        
        # Resize embeddings if special tokens were added
        if use_persona:
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Load LoRA adapter if checkpoint provided
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"Loading LoRA adapter from: {checkpoint_path}")
            self.model = PeftModel.from_pretrained(self.model, checkpoint_path)
            print("LoRA adapter loaded successfully!")
        elif checkpoint_path:
            print(f"Warning: Checkpoint not found at {checkpoint_path}, using base model only")
        
        self.model.eval()
        print("Model loaded and ready for chat!\n")
    
    def generate_response(
        self,
        user_input: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
    ) -> str:
        """Generate a response to user input."""
        
        # Build prompt with conversation history
        prompt = self._build_prompt(user_input)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device != "auto":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Set stop tokens
        eos_token_ids = [self.tokenizer.eos_token_id]
        if self.use_persona:
            # Also stop at </user1> token
            user1_end_id = self.tokenizer.convert_tokens_to_ids(USER1_END)
            if user1_end_id != self.tokenizer.unk_token_id:
                eos_token_ids.append(user1_end_id)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=eos_token_ids,
            )
        
        # Decode only the new tokens
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=False if self.use_persona else True
        )
        
        # Clean up response for persona mode (remove end token)
        if self.use_persona:
            response = response.replace(USER1_END, "").strip()
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response.strip()
    
    def _build_prompt(self, user_input: str) -> str:
        """Build prompt with system message and conversation history."""
        
        # Persona mode: use special token format
        if self.use_persona and self.persona:
            parts = []
            
            # Add persona with special tokens
            parts.append(f"{PERSONA_START}{self.persona}{PERSONA_END}")
            
            # Conversation history with special tokens (last 5 exchanges)
            for msg in self.conversation_history[-10:]:
                if msg["role"] == "user":
                    parts.append(f"{USER2_START}{msg['content']}{USER2_END}")
                else:
                    parts.append(f"{USER1_START}{msg['content']}{USER1_END}")
            
            # Current user input (User 2 is the human)
            parts.append(f"{USER2_START}{user_input}{USER2_END}")
            # Start User 1 response (model will complete)
            parts.append(f"{USER1_START}")
            
            return "".join(parts)
        
        # Standard mode: system prompt + conversation
        parts = []
        
        # System prompt
        if self.system_prompt:
            parts.append(f"System: {self.system_prompt}\n")
        
        # Conversation history (last 5 exchanges to avoid context overflow)
        for msg in self.conversation_history[-10:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            parts.append(f"{role}: {msg['content']}")
        
        # Current user input
        parts.append(f"User: {user_input}")
        parts.append("Assistant:")
        
        return "\n".join(parts)
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        print("Conversation history cleared.")
    
    def set_persona(self, persona: str):
        """Set the persona for persona-conditioned generation."""
        self.persona = persona.strip()
        print(f"Persona set: {self.persona[:100]}{'...' if len(self.persona) > 100 else ''}")
    
    def chat_loop(self):
        """Interactive chat loop."""
        print("=" * 60)
        print("Welcome to the Chat Interface!")
        print("=" * 60)
        print("Commands:")
        print("  /clear  - Clear conversation history")
        if self.use_persona:
            print("  /persona <text> - Change persona")
        else:
            print("  /system <text> - Set system prompt")
        print("  /temp <value> - Set temperature (0.0-2.0)")
        print("  /quit or /exit - Exit chat")
        print("=" * 60)
        
        # Prompt for persona if in persona mode
        if self.use_persona:
            print("\nPersona-conditioned mode active.")
            
            # Use default persona if set, otherwise prompt
            if DEFAULT_PERSONA:
                self.set_persona(DEFAULT_PERSONA)
                print(f"Using default persona: {self.persona}")
            else:
                print("Enter your persona description (who is the AI?):")
                print("Example: I love hiking. I work as a nurse. I have two cats.")
                print()
                while not self.persona:
                    persona_input = input("Persona: ").strip()
                    if persona_input:
                        self.set_persona(persona_input)
                print()
                print(f"Great! The AI will now respond as: {self.persona}")
        elif self.system_prompt:
            print(f"System: {self.system_prompt}")
        print()
        
        temperature = 0.7
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    cmd = user_input.lower()
                    
                    if cmd in ["/quit", "/exit", "/q"]:
                        print("Goodbye!")
                        break
                    elif cmd == "/clear":
                        self.clear_history()
                        continue
                    elif cmd.startswith("/system "):
                        self.system_prompt = user_input[8:]
                        print(f"System prompt set to: {self.system_prompt}")
                        continue
                    elif cmd.startswith("/persona "):
                        self.set_persona(user_input[9:])
                        self.clear_history()  # Clear history when changing persona
                        continue
                    elif cmd.startswith("/temp "):
                        try:
                            temperature = float(cmd[6:])
                            print(f"Temperature set to: {temperature}")
                        except ValueError:
                            print("Invalid temperature value")
                        continue
                    else:
                        print(f"Unknown command: {cmd}")
                        continue
                
                # Generate response
                response = self.generate_response(
                    user_input,
                    temperature=temperature,
                )
                print(f"Assistant: {response}\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Chat with fine-tuned language models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python app/chat.py persona        # Persona-conditioned chat
    python app/chat.py 19thcentury    # 19th century literature style
    python app/chat.py standard       # Base model (no fine-tuning)
    python app/chat.py persona --4bit # Persona with 4-bit quantization
        """
    )
    parser.add_argument(
        "mode", type=str, nargs="?", default="standard",
        choices=list(MODEL_CONFIGS.keys()),
        help="Which model mode to use: persona, 19thcentury, or standard (default: standard)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Custom checkpoint path (overrides mode default)"
    )
    parser.add_argument(
        "--base-model", type=str, default=None,
        help="Custom base model (overrides mode default)"
    )
    parser.add_argument(
        "--system-prompt", type=str, default=None,
        help="Custom system prompt"
    )
    parser.add_argument(
        "--8bit", dest="load_8bit", action="store_true",
        help="Load model in 8-bit quantization"
    )
    parser.add_argument(
        "--4bit", dest="load_4bit", action="store_true",
        help="Load model in 4-bit quantization"
    )
    args = parser.parse_args()
    
    # Get config for selected mode
    config = MODEL_CONFIGS[args.mode]
    
    # Apply overrides
    base_model = args.base_model or config["base_model"]
    checkpoint = args.checkpoint if args.checkpoint is not None else config["checkpoint"]
    system_prompt = args.system_prompt if args.system_prompt is not None else config.get("system_prompt", "")
    use_persona = config.get("use_persona", False)
    
    print(f"Mode: {args.mode}")
    print(f"Base model: {base_model}")
    print(f"Checkpoint: {checkpoint or '(none - using base model)'}")
    print()
    
    # Create chatbot
    bot = ChatBot(
        base_model=base_model,
        checkpoint_path=checkpoint,
        system_prompt=system_prompt,
        load_in_8bit=args.load_8bit,
        load_in_4bit=args.load_4bit,
        use_persona=use_persona,
    )
    
    # Start chat loop
    bot.chat_loop()


if __name__ == "__main__":
    main()
