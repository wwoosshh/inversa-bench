"""Explicit, tier-balanced ~100-model leaderboard roster. Hand-picked across families and
capability tiers (weak -> frontier); intersected with the live OpenRouter catalog so only
reachable-by-listing slugs survive. Writes data/banks/leaderboard_roster.json.
"""
import json

PICKS = [
 # OpenAI
 "openai/gpt-4o-mini","openai/gpt-4o","openai/gpt-4.1-mini","openai/gpt-4.1","openai/gpt-5-mini",
 "openai/gpt-5","openai/gpt-5.1","openai/gpt-5.2","openai/gpt-oss-20b","openai/gpt-oss-120b",
 "openai/o4-mini","openai/o3-mini",
 # Qwen
 "qwen/qwen-2.5-7b-instruct","qwen/qwen-2.5-72b-instruct","qwen/qwen3-8b","qwen/qwen3-14b",
 "qwen/qwen3-32b","qwen/qwen3-30b-a3b","qwen/qwen3-235b-a22b","qwen/qwen3-235b-a22b-thinking-2507",
 "qwen/qwen3-max","qwen/qwen3.5-plus-02-15","qwen/qwen3.7-plus",
 # Anthropic
 "anthropic/claude-3-haiku","anthropic/claude-3.5-haiku","anthropic/claude-haiku-4.5",
 "anthropic/claude-sonnet-4","anthropic/claude-sonnet-4.5","anthropic/claude-opus-4",
 "anthropic/claude-opus-4.1","anthropic/claude-opus-4.5","anthropic/claude-opus-4.8",
 # Google
 "google/gemini-2.5-flash-lite","google/gemini-2.5-flash","google/gemini-2.5-pro",
 "google/gemini-3.1-flash-lite","google/gemini-3.5-flash","google/gemma-3-4b-it",
 "google/gemma-3-12b-it","google/gemma-3-27b-it",
 # Mistral
 "mistralai/mistral-nemo","mistralai/ministral-8b-2512","mistralai/mistral-small-2603",
 "mistralai/mistral-small-3.2-24b-instruct","mistralai/mistral-medium-3","mistralai/mistral-large",
 "mistralai/mixtral-8x22b-instruct",
 # DeepSeek
 "deepseek/deepseek-chat","deepseek/deepseek-chat-v3.1","deepseek/deepseek-v3.2",
 "deepseek/deepseek-r1","deepseek/deepseek-r1-0528","deepseek/deepseek-v4-pro",
 # z-ai (GLM)
 "z-ai/glm-4-32b","z-ai/glm-4.5-air","z-ai/glm-4.5","z-ai/glm-4.6","z-ai/glm-4.7","z-ai/glm-5",
 # Meta
 "meta-llama/llama-3.1-8b-instruct","meta-llama/llama-3.1-70b-instruct",
 "meta-llama/llama-3.3-70b-instruct","meta-llama/llama-4-scout","meta-llama/llama-4-maverick",
 # MiniMax
 "minimax/minimax-01","minimax/minimax-m1","minimax/minimax-m2","minimax/minimax-m2.5",
 # xAI
 "x-ai/grok-4.20","x-ai/grok-4.3","x-ai/grok-build-0.1",
 # Moonshot
 "moonshotai/kimi-k2","moonshotai/kimi-k2-thinking","moonshotai/kimi-k2.6",
 # Cohere
 "cohere/command-r-08-2024","cohere/command-r-plus-08-2024","cohere/command-a",
 # Amazon
 "amazon/nova-lite-v1","amazon/nova-pro-v1","amazon/nova-premier-v1",
 # Nvidia
 "nvidia/llama-3.3-nemotron-super-49b-v1.5","nvidia/nemotron-3-ultra-550b-a55b",
 # Nous
 "nousresearch/hermes-3-llama-3.1-70b","nousresearch/hermes-4-70b",
 # Microsoft
 "microsoft/phi-4","microsoft/phi-4-mini-instruct",
 # ByteDance
 "bytedance-seed/seed-1.6","bytedance-seed/seed-2.0-mini",
 # Xiaomi
 "xiaomi/mimo-v2.5","xiaomi/mimo-v2.5-pro",
 # Arcee
 "arcee-ai/virtuoso-large","arcee-ai/maestro-reasoning",
 # singletons / long tail
 "ai21/jamba-large-1.7","ibm-granite/granite-4.1-8b","upstage/solar-pro-3","writer/palmyra-x5",
 "allenai/olmo-3-32b-think","liquid/lfm-2-24b-a2b","inception/mercury-2",
 "tencent/hunyuan-a13b-instruct","inflection/inflection-3-productivity","stepfun/step-3.7-flash",
]

catalog = set(json.load(open("data/results/_openrouter_catalog.json")))
keep = [m for m in PICKS if m in catalog]
drop = [m for m in PICKS if m not in catalog]
json.dump(keep, open("data/banks/leaderboard_roster.json", "w"), indent=1)
print(f"roster: {len(keep)} models (picked {len(PICKS)}, {len(drop)} not in live catalog)")
if drop:
    print("not found (dropped):", drop)
