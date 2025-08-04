# Natural Language Datetime Conversion Fine-tuning

A self-learning project that fine-tunes a language model to convert natural language time expressions into structured datetime formats using LoRA (Low-Rank Adaptation) with Unsloth.

## 🎯 Project Overview

This project demonstrates how to fine-tune Llama-3.2-3B-Instruct to accurately convert human-readable time expressions like "tomorrow at 3 PM" or "next Friday evening" into structured JSON datetime objects with precise ISO 8601 timestamps.

### Example Conversion
```
Input: "tomorrow at 3 PM"
Output: {
  "start": "2025-08-06T15:00:00",
  "end": "2025-08-06T15:00:00"
}
```
## Requirements
- Get wandb API from here: https://wandb.ai/authorize?ref=models

## 🚀 Features

- **Natural Language Processing**: Handles diverse time expressions including relative dates, holidays, seasons, and colloquial terms
- **Structured Output**: Generates JSON with start/end datetime pairs in ISO 8601 format
- **Memory Efficient**: Uses LoRA fine-tuning with 4-bit quantization for reduced memory usage
- **Comprehensive Dataset**: Includes 150+ synthetic examples covering various temporal expressions

## 📊 Dataset(Synthetic data)

The training dataset contains diverse temporal expressions including:

- **Relative Times**: "in 15 minutes", "3 days ago", "next week"
- **Specific Dates**: "March 15th 2025", "tomorrow at noon"
- **Holidays**: "Christmas Day", "next Thanksgiving", "Halloween night"
- **Seasonal References**: "start of summer", "last winter"
- **Business Terms**: "next quarter", "end of fiscal year"
- **Natural Descriptions**: "dawn", "dusk", "golden hour", "witching hour"

### Data Format
Each line in `paste.txt` follows this JSON structure:
```json
{
  "instruction": "Convert to datetime",
  "input": "tomorrow at 3 PM",
  "output": {
    "start": "2025-08-06T15:00:00",
    "end": "2025-08-06T15:00:00"
  }
}
```

## 🔧 Usage

### 1. Prepare Data
Ensure your training data is in `time_parsing_dataset.jsonl` with the correct JSON format.

### 2. Run Training
```bash
python finetune_datetime.py
```

### 3. Training Parameters
- **Model**: Llama-3.2-3B-Instruct
- **LoRA Rank**: 16
- **Batch Size**: 2 (with 4x gradient accumulation)
- **Learning Rate**: 2e-4
- **Max Steps**: 100
- **Sequence Length**: 2048 tokens

### 4. Test the Model
The script automatically tests the fine-tuned model with sample inputs:
```python
test_prompts = [
    "tomorrow at 3 PM",
    "next Friday evening", 
    "in 2 hours",
    "last Monday morning",
    "Christmas Day 2025"
]
```

## 📈 Training Process

1. **Data Loading**: Parses JSON lines from text file
2. **Conversation Formatting**: Converts to chat format for instruction tuning
3. **LoRA Fine-tuning**: Applies low-rank adaptation to specific model layers
4. **Inference Testing**: Validates model performance on test cases

## 🎓 Self-Learning Objectives

This project serves as a hands-on learning experience for:

- **Language Model Fine-tuning**: Understanding LoRA and parameter-efficient training
- **Instruction Tuning**: Creating conversational datasets for specific tasks
- **Temporal Reasoning**: Teaching models to understand and process time expressions
- **Synthetic Data Generation**: Creating diverse training examples programmatically
- **Model Evaluation**: Testing fine-tuned models on domain-specific tasks

## 🔍 Key Learning Points

### Technical Skills
- Implementing LoRA fine-tuning with Unsloth
- Working with Hugging Face transformers and datasets
- Managing GPU memory with quantization techniques
- Designing effective chat templates for instruction following

## 📊 Expected Results

After fine-tuning, the model should accurately convert:
- ✅ Relative time expressions ("in 2 hours", "yesterday")
- ✅ Absolute dates ("March 15th", "Christmas 2025")
- ✅ Time ranges ("from 9 AM to 5 PM")
- ✅ Natural language times ("dawn", "noon", "midnight")
- ✅ Complex expressions ("the third Thursday of next month")

## 🚧 Challenges & Solutions

### Challenge: Ambiguous Time References
**Solution**: Use consistent reference point (current datetime) and clear context rules

### Challenge: Memory Limitations
**Solution**: LoRA fine-tuning with 4-bit quantization reduces memory usage by ~75%

### Challenge: Limited Training Data
**Solution**: Generate comprehensive synthetic dataset covering edge cases
