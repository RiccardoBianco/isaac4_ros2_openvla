from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Carica il modello Flan-T5 base (puoi usare anche flan-t5-large o flan-t5-xl se vuoi più potenza)
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Istruzione complessa
instruction = "Put the green cube on the red area, then put the blue cube on the red area, and finally put the yellow cube on the red area."

prompt = (
    "Break down each instruction into simple, numbered steps.\n\n"
    "Instruction: Put the box on the red area, then put the cup on the red area, and finally put the spoon on the red area.\n"
    "Steps:"
    "1. Put the box on the red area.\n"
    "2. Put the cup on the red area.\n"
    "3. Put the spoon on the red area.\n\n"
    "Instruction: Put the ball in the blue area, then put the bottle in the blue area, and finally Put the fork in the blue area.\n"
    "Steps:"
    "1. Put the ball in the blue area.\n"
    "2. Put the bottle in the blue area.\n"
    "3. Put the fork in the blue area.\n\n"
    "Instruction: Put the fork on the red area, then put the phtone on the red area, and finally Put the paper bin on the red area.\n"
    "Steps:"
    "1. Put the fork on the red area.\n"
    "2. Put the phone on the red area.\n"
    "3. Put the paper bin on the red area.\n\n"
    f"Instruction: {instruction}\n"
    "Steps:"  # Prompt to generate the steps for the new instruction
)


# Tokenizza e genera l'output
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=300)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Decomposed steps:")
print(result)

import re
parts = re.split(r'\d+\.\s*', result)
# Rimuove la prima parte vuota (prima del "1.")
instructions = [p.strip().rstrip('.') for p in parts if p]

print(instructions)


