import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
import joblib
import os

def extrair_parametros(param_string):
    param_dict = {}
    for item in param_string.split(", "):
        key, value = item.split(": ")
        param_dict[key.strip()] = float(value)
    return param_dict

def converter_para_coordenadas(posicao):
    return list(map(float, posicao.strip("()").split(", ")))

def calcular_setor(delta_x, delta_y, limite_diagonal=0.5):
    if delta_x < 0 and abs(delta_y / delta_x) <= limite_diagonal:
        return 2 
    elif delta_y > 0:
        return 3 
    elif delta_y < 0:
        return 1  
    else:
        return 0
    
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 5000
LATENT_DIM = 10
COND_DIM = 3
PARAM_DIM = 4 
LEARNING_RATE = 0.0002

engine = create_engine("mysql+mysqlconnector://lara:xxxx@127.0.0.1/simulation_db")
query = "SELECT * FROM base_resultados"
df = pd.read_sql(query, engine)

df["initial_coords"] = df["initial_position"].apply(converter_para_coordenadas)
df["final_coords"] = df["final_position"].apply(converter_para_coordenadas)
df["delta_x"] = df["final_coords"].apply(lambda x: x[0]) - df["initial_coords"].apply(lambda x: x[0])
df["delta_y"] = df["final_coords"].apply(lambda x: x[1]) - df["initial_coords"].apply(lambda x: x[1])
df["setor"] = df.apply(lambda row: calcular_setor(row["delta_x"], row["delta_y"]), axis=1)

dados_aceitaveis = df[df["result"] == "aceitável"]
parametros_expandido = dados_aceitaveis["parameters"].apply(extrair_parametros).apply(pd.Series)
df_completo = pd.concat([dados_aceitaveis, parametros_expandido], axis=1)

param_cols = ["alpha_i", "alpha_h", "k", "delta_phi_vh"]
scaler = StandardScaler()
df_completo[param_cols] = scaler.fit_transform(df_completo[param_cols])

class ParametrosDataset(Dataset):
    def __init__(self, df, param_cols):
        self.parametros = df[param_cols].values.astype(np.float32)
        self.setores = df["setor"].values.astype(int) - 1 

    def __len__(self):
        return len(self.parametros)

    def __getitem__(self, idx):
        return torch.tensor(self.parametros[idx]), torch.tensor(self.setores[idx])

dataset = ParametrosDataset(df_completo, param_cols)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

class Generator(nn.Module):
    def __init__(self, latent_dim, cond_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, noise, cond):
        x = torch.cat([noise, cond], dim=1)
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim, cond_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + cond_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, cond):
        x = torch.cat([x, cond], dim=1)
        return self.model(x)

G = Generator(LATENT_DIM, COND_DIM, PARAM_DIM).to(DEVICE)
D = Discriminator(PARAM_DIM, COND_DIM).to(DEVICE)

optimizer_G = optim.Adam(G.parameters(), lr=LEARNING_RATE)
optimizer_D = optim.Adam(D.parameters(), lr=LEARNING_RATE)
criterion = nn.BCELoss()

print("Iniciando treinamento do CGAN...")
for epoch in range(EPOCHS):
    for real_data, labels in loader:
        real_data, labels = real_data.to(DEVICE), labels.to(DEVICE)
        batch_size = real_data.size(0)

        cond = torch.nn.functional.one_hot(labels, num_classes=COND_DIM).float()

        real_labels = torch.ones(batch_size, 1).to(DEVICE)
        fake_labels = torch.zeros(batch_size, 1).to(DEVICE)

        noise = torch.randn(batch_size, LATENT_DIM).to(DEVICE)
        fake_data = G(noise, cond)
        real_output = D(real_data, cond)
        fake_output = D(fake_data.detach(), cond)

        loss_D = criterion(real_output, real_labels) + criterion(fake_output, fake_labels)
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        fake_output = D(fake_data, cond)
        loss_G = criterion(fake_output, real_labels)
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

    if (epoch + 1) % 500 == 0:
        print(f"[{epoch+1}/{EPOCHS}] Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")

os.makedirs("modelos_cgan", exist_ok=True)
torch.save(G.state_dict(), "modelos_cgan/gerador.pth")
torch.save(D.state_dict(), "modelos_cgan/discriminador.pth")
joblib.dump(scaler, "modelos_cgan/scaler.pkl")
