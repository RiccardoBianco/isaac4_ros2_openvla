import os

# Imposta il percorso della cartella
folder = './isaac_ws/src/output/episodes'  # Sostituisci con il percorso corretto

# # Prima parte: 3001–5000 → 4001–6000 (sposta prima questa per evitare conflitti)
# for i in range(5000, 3000, -1):  # ordine decrescente!
#     old_name = f"episode_{i:04d}.npy"
#     new_index = i + 1000
#     new_name = f"episode_{new_index:04d}.npy"
    
#     old_path = os.path.join(folder, old_name)
#     new_path = os.path.join(folder, new_name)

#     if os.path.exists(old_path):
#         os.rename(old_path, new_path)
#         print(f"Renamed {old_name} → {new_name}")
#     else:
#         print(f"File {old_name} not found")

# # Seconda parte: 0001–3000 → 6001–9000
# for i in range(3000, 0, -1):  # anche qui meglio decrescente per sicurezza
#     old_name = f"episode_{i:04d}.npy"
#     new_index = i + 6000
#     new_name = f"episode_{new_index:04d}.npy"
    
#     old_path = os.path.join(folder, old_name)
#     new_path = os.path.join(folder, new_name)

#     if os.path.exists(old_path):
#         os.rename(old_path, new_path)
#         print(f"Renamed {old_name} → {new_name}")
#     else:
#         print(f"File {old_name} not found")

# 9001-9250 → 3751–4000 # TODO check with chatty
for i in range(9250, 9000, -1):  # anche qui meglio decrescente per sicurezza
    old_name = f"episode_{i:04d}.npy"
    new_index = i + 5250
    new_name = f"episode_{new_index:04d}.npy"
    
    old_path = os.path.join(folder, old_name)
    new_path = os.path.join(folder, new_name)

    if os.path.exists(old_path):
        os.rename(old_path, new_path)
        print(f"Renamed {old_name} → {new_name}")
    else:
        print(f"File {old_name} not found")