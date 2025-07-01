from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Carica il modello Flan-T5 base (puoi usare anche flan-t5-large o flan-t5-xl se vuoi più potenza)
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Istruzione complessa
instruction = "Put the green cube on the red area, then put the blue cube on the red area, and finally put the yellow cube on the red area."

instruction = (
    "Place the cracker_box in the brown box.\n"
    "Then, place the tomato_can in the brown box.\n"
    "Finally, place the school_bus in the brown box.\n"
)

prompt = (
    "Break down each instruction into simple, numbered steps.\n\n"
    "Instruction: Place the box in the brown box, then place the cup in the brown box, and finally place the spoon in the brown box.\n"
    "Steps:\n"
    "1. Place the box in the brown box.\n"
    "2. Place the cup in the brown box.\n"
    "3. Place the spoon in the brown box.\n\n"
    "Instruction: Place the ball in the brown box, then place the bottle in the brown box, and finally place the fork in the brown box.\n"
    "Steps:\n"
    "1. Place the ball in the brown box.\n"
    "2. Place the bottle in the brown box.\n"
    "3. Place the fork in the brown box.\n\n"
    "Instruction: Place the fork in the brown box, then place the phone in the brown box, and finally place the paper bin in the brown box.\n"
    "Steps:\n"
    "1. Place the fork in the brown box.\n"
    "2. Place the phone in the brown box.\n"
    "3. Place the paper bin in the brown box.\n\n"
    f"Instruction: {instruction}"
    "Steps:"
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


