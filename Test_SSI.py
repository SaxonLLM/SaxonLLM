from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import torch
import time

start_time = time.time()

# Loading the Model and LoRA
model_name = "meta-llama/Llama-3.2-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token 

adapter_path = "./SSI"
lora_config = LoraConfig.from_pretrained(adapter_path)
model = get_peft_model(model, lora_config)
model = model.to("cuda")

load_time = time.time() - start_time
print(f"Model loaded in {load_time:.2f} seconds")

# Prompts
prompts = [
    "You are a Sächsisch storywriter who only writes in Sächsisch. You don't speak any English. You don't speak Hochdeutsch. Donot answer in English and Donot answer in Hochdeutsch. Answer in Sächsisch: Write a short story about two Saxons in the Deutsche Bahn.",
    "You are a English to Sächsisch translator who only writes in Sächsisch.You don't speak any English. You don't speak Hochdeutsch. Donot answer in English and Donot answer in Hochdeutsch. Translate to Sächsisch:: 'Our team consists of more than 180 people, including renowned international researchers as well as highly skilled professionals in administrative and communicative roles. With more than 60 principal investigators, two Humboldt Professorships and up to twelve planned AI Professorships we support excellence in research and teaching in Leipzig and Dresden. Promoting young talent is also an important part of our work, therefore we have established four Junior Research Groups that meaningfully complement our current research topics. Furthermore, we are welcoming Associated Members who contribute their expertise to our center.'",
    "You are a Sächsisch historian who only writes in Sächsisch.You don't speak any English. You don't speak Hochdeutsch. Donot answer in English and Donot answer in Hochdeutsch. Answer in Sächsisch: What is the history of TU Dresden?",
    "Answer in Sächsisch: Write a short story about two Saxons in the Deutsche Bahn.",
    "Translate to Sächsisch: 'Our team consists of more than 180 people, including renowned international researchers as well as highly skilled professionals in administrative and communicative roles. With more than 60 principal investigators, two Humboldt Professorships and up to twelve planned AI Professorships we support excellence in research and teaching in Leipzig and Dresden. Promoting young talent is also an important part of our work, therefore we have established four Junior Research Groups that meaningfully complement our current research topics. Furthermore, we are welcoming Associated Members who contribute their expertise to our center.'",
    "Answer in Sächsisch: What is the history of TU Dresden?",
    "How do you say 'Good Morning' in Sächsisch?"
]

repeated_prompts = prompts * 5 

response_times = []
responses = []

for i, prompt in enumerate(repeated_prompts):
    start_response_time = time.time()
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")
    
    # Generate the output
    with torch.autocast("cuda"):
        output = model.generate(**inputs, max_new_tokens=1000)
    
    # Record time and response
    response_time = time.time() - start_response_time
    response_times.append(response_time)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    responses.append(response)
    
    print(f"Response for Prompt {i+1} generated in {response_time:.2f} seconds")
    print("-" * 50)

# Saving
with open("SSI.txt", "w") as f:
    for i, (response, time_taken) in enumerate(zip(responses, response_times)):
        f.write(f"Prompt {i+1}:\n")
        f.write(f"Response Time: {time_taken:.2f} seconds\n")
        f.write(f"Response:\n{response}\n")
        f.write("-" * 50 + "\n")

total_generation_time = sum(response_times)
print(f"Total generation time: {total_generation_time:.2f} seconds")
print(f"Average time per response: {total_generation_time / len(response_times):.2f} seconds")
