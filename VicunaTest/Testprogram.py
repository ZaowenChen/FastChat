import torch
import argparse
from fastchat.model import load_model
from fastchat.conversation import get_conv_template
from transformers import AutoTokenizer

def safe_load_tokenizer(pretrained_model_name_or_path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        return tokenizer
    except Exception as e:
        print(f"Failed to load tokenizer from {pretrained_model_name_or_path}: {str(e)}")
        return None

def batch_process(prompts, args):
    model, tokenizer = load_model(
        args.vicuna_dir,
        args.device,
        args.num_gpus,
        args.max_gpu_memory,
        args.load_8bit,
        debug=args.debug,
    )

    conv_template = get_conv_template("vicuna_v1.1")
    
    # Prepare batch prompts
    batch_prompts = [conv_template.get_prompt_with_user_message(prompt) for prompt in prompts]

    # Tokenize the batch prompts
    input_ids = tokenizer(batch_prompts).input_ids

    # Generate responses in batch
    outputs_ids = model.generate(
        torch.as_tensor(input_ids).to(args.device),
        do_sample=True,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    # Decode the batch of outputs
    responses = [tokenizer.decode(output_ids[i], skip_special_tokens=True, spaces_between_special_tokens=False) for i in range(len(outputs_ids))]

    return responses

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vicuna-dir", type=str, default="/home/zc8vc/FastChat/vicuna-13b-v1.5", help="The path to the weights")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default="cuda")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--max-gpu-memory", type=str, help="The maximum memory per gpu. Use a string like '13Gib'")
    parser.add_argument("--load-8bit", action="store_true", help="Use 8-bit quantization.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Example prompts for batch processing
    example_prompts = ["What is the weather today?", "How does quantum computing work?", "Tell me a joke."]
    responses = batch_process(example_prompts, args)
    for response in responses:
        print("ASSISTANT:", response)
