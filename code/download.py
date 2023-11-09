from transformers import AutoModelForCausalLM, AutoTokenizer

# Name of the model
model_name = "TheBloke/Llama-2-13B-chat-GPTQ"

# Load pre-trained model tokenizer (vocabulary)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Download and cache the pre-trained model weights
model = AutoModelForCausalLM.from_pretrained(model_name)

# Save the model and tokenizer to a specified directory
model.save_pretrained('./specify-your-directory/')
tokenizer.save_pretrained('./specify-your-directory/')