import os
from unsloth import FastLanguageModel
from inference.agent_loop import AgentLoop, make_hf_generate_fn
from tools import dispatch

def run_stage_test(adapter_path, question):
    print("\n" + "="*60)
    print(f"LOADING ADAPTER: {adapter_path}")
    print("="*60)
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path, 
        max_seq_length=1024,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)    
    hf_gen_fn = make_hf_generate_fn(model, tokenizer)
    agent = AgentLoop(generate_fn=hf_gen_fn, tool_dispatch_fn=dispatch)
    
    print(f"\nPrompt: {question}\n")
    print("Running Agent Loop...\n")    
    result = agent.run(question)
    print("\n" + "-"*30 + " FINAL TRAJECTORY " + "-"*30)
    print(result.full_trajectory)
    print("-" * 78)
    
    del model
    del tokenizer
    import torch
    torch.cuda.empty_cache()

if __name__ == "__main__":
    test_prompt = "A company has a budget of $5000. They spend 23% on marketing and 15% on legal. How much money do they have left? Use your tools to calculate precisely."    
    stages = {
        "1_reasoning": "./outputs/sft_reasoning",
        "2_combined": "./outputs/sft_combined",
        "3_grpo": "./outputs/grpo_lora"
    }
    
    selected_stage = "3_grpo" 
    if os.path.exists(stages[selected_stage]):
        run_stage_test(stages[selected_stage], test_prompt)
    else:
        print(f"Path not found: {stages[selected_stage]}. Make sure training completed successfully.")