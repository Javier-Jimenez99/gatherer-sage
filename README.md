# gatherer-sage
AI chatbot to answer MTG ruling questions

## Sources
### Datasets
- MTG rules: 
  - Magic Comprehensive Rules: [https://media.wizards.com/2024/downloads/MagicCompRules%2004102024.txt](https://media.wizards.com/2024/downloads/MagicCompRules%2004102024.txt)
  - Magic Tournament Rules: [https://media.wizards.com/ContentResources/WPN/MTG_MTR_2024_May13.pdf](https://media.wizards.com/ContentResources/WPN/MTG_MTR_2024_May13.pdf)
  - Magic Infraction Procedure Guide: [https://media.wizards.com/ContentResources/WPN/MTG_IPG_2024Apr15_EN.pdf](https://media.wizards.com/ContentResources/WPN/MTG_IPG_2024Apr15_EN.pdf)
  - Digital Infraction Procedure Guide: [https://assets.ctfassets.net/ryplwhabvmmk/7ndnTO3hk658elrr77KJHC/b0de8c6a2d41a3b2934c14d35dc6d40d/Magic_Digital_IPG_Update_11152021.pdf](https://assets.ctfassets.net/ryplwhabvmmk/7ndnTO3hk658elrr77KJHC/b0de8c6a2d41a3b2934c14d35dc6d40d/Magic_Digital_IPG_Update_11152021.pdf)
  - Digital Tournament Rules: [https://assets.ctfassets.net/ryplwhabvmmk/5HYQjgeT4YBN7pvVZk2hyY/d40ebd6f80057637df726398ba2bb72c/Magic_Digital_Tournament_Rules_Update_11152021.pdf](https://assets.ctfassets.net/ryplwhabvmmk/5HYQjgeT4YBN7pvVZk2hyY/d40ebd6f80057637df726398ba2bb72c/Magic_Digital_Tournament_Rules_Update_11152021.pdf)
  - Judging at Regular Rules Enforcement Level: [https://media.wizards.com/2022/wpn/marketing_materials/wpn/mtg_jar_25sep20_en.pdf](https://media.wizards.com/2022/wpn/marketing_materials/wpn/mtg_jar_25sep20_en.pdf)
- Ruling Subreddit: 
  - [https://www.reddit.com/r/mtgrules/](https://www.reddit.com/r/mtgrules/)
  - [https://www.reddit.com/r/askajudge](https://www.reddit.com/r/askajudge)
- MTG Salvation Forum: [https://www.mtgsalvation.com/forums/magic-fundamentals/magic-rulings](https://www.mtgsalvation.com/forums/magic-fundamentals/magic-rulings)
- Rules Q&A:[https://rulesguru.net/](https://rulesguru.net/)

### Data Utils
- Download Subreddit: [https://pushshift.io/signup](https://pushshift.io/signup)
- Hyperlink MTG Rules: [https://yawgatog.com/resources/magic-rules/](https://yawgatog.com/resources/magic-rules/)
- MTG list of rules docs: [https://blogs.magicjudges.org/o/rules-policy-documents/](https://blogs.magicjudges.org/o/rules-policy-documents/)

### Models
- Llama 3: [8B-instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) and [70B-instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)
- Mistral: [7B-instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)

### Models Utils
- LangChain RAG example: [https://huggingface.co/learn/cookbook/advanced_rag](https://huggingface.co/learn/cookbook/advanced_rag)
- VRAM requirements: [https://www.reddit.com/r/LocalLLaMA/comments/18o5u0k/helpful_vram_requirement_table_for_qlora_lora_and/](https://www.reddit.com/r/LocalLLaMA/comments/18o5u0k/helpful_vram_requirement_table_for_qlora_lora_and/)
- QLora example: [https://dassum.medium.com/fine-tune-large-language-model-llm-on-a-custom-dataset-with-qlora-fb60abdeba07](https://dassum.medium.com/fine-tune-large-language-model-llm-on-a-custom-dataset-with-qlora-fb60abdeba07)
- Fine-tuning example: [https://github.com/huggingface/blog/blob/main/Lora-for-sequence-classification-with-Roberta-Llama-Mistral.md](https://github.com/huggingface/blog/blob/main/Lora-for-sequence-classification-with-Roberta-Llama-Mistral.md)

## Setup

Installation Advices:
```
FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip install flash-attn --no-build-isolation`
```
