"""Short on-policy GRPO pass using exact archived tier scores."""

import json

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from run import ADAPTER, MODEL, ROOT, norm, prompt_text


def reward(completions, answers, **kwargs):
    scores = []
    for completion, sheet in zip(completions, answers):
        text = completion if isinstance(completion, str) else completion[0]["content"]
        answer = text.strip().split("\n")[0].strip()
        lookup = {norm(a): score for a, score in json.loads(sheet).items()}
        scores.append(lookup.get(norm(answer), 0) / 100)
    return scores


def main():
    if not torch.backends.mps.is_available():
        raise RuntimeError("This run needs MPS; use the approved GPU command.")
    rows = [r for r in json.loads((ROOT / "data" / "dives.json").read_text()) if r["split"] == "train"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, dtype=torch.float32)
    model = PeftModel.from_pretrained(base, ADAPTER, is_trainable=True)
    dataset = Dataset.from_list([{"prompt": prompt_text(tokenizer, r["prompt"]),
                                  "answers": json.dumps(r["answers"], ensure_ascii=False)} for r in rows])
    config = GRPOConfig(output_dir=str(ROOT / "results" / "grpo_logs"),
                        per_device_train_batch_size=4, num_generations=4,
                        max_completion_length=24, max_steps=25,
                        learning_rate=1e-5, beta=0.02,
                        gradient_checkpointing=False, optim="adamw_torch",
                        logging_steps=5, save_strategy="no", report_to="none",
                        disable_tqdm=True, bf16=False, fp16=False)
    trainer = GRPOTrainer(model=model, reward_funcs=reward, args=config,
                          train_dataset=dataset, processing_class=tokenizer)
    trainer.train()
    out = ROOT / "results" / "grpo_lora"
    trainer.model.save_pretrained(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
