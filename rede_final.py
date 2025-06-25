import torch
import torch.nn as nn
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# --- CONFIGURAÇÕES ---
LATENT_DIM = 10
COND_DIM = 3
PARAM_DIM = 4
NUM_AMOSTRAS = 66  # quantas amostras gerar por setor
SETOR_DESEJADO = 3  # 1: Direita, 2: Frente, 3: Esquerda
DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

G = Generator(LATENT_DIM, COND_DIM, PARAM_DIM).to(DISPOSITIVO)
G.load_state_dict(torch.load("modelos_cgan/gerador.pth", map_location=DISPOSITIVO))
G.eval()

scaler = joblib.load("modelos_cgan/scaler.pkl")

cond_vec = torch.nn.functional.one_hot(
    torch.tensor([SETOR_DESEJADO - 1] * NUM_AMOSTRAS),
    num_classes=COND_DIM
).float().to(DISPOSITIVO)

noise = torch.randn(NUM_AMOSTRAS, LATENT_DIM).to(DISPOSITIVO)
with torch.no_grad():
    parametros_gerados_norm = G(noise, cond_vec).cpu().numpy()

parametros_gerados = scaler.inverse_transform(parametros_gerados_norm)
df_resultado = pd.DataFrame(parametros_gerados, columns=["alpha_i", "alpha_h", "k", "delta_phi_vh"])
df_resultado["setor"] = SETOR_DESEJADO

df_resultado["k"] = df_resultado["k"].round().astype(int)
df_resultado = df_resultado[(df_resultado >= 0).all(axis=1)].reset_index(drop=True)

try:
    mlp = joblib.load("modelo_mlp.pkl")
    predicoes = mlp.predict(df_resultado[["alpha_i", "alpha_h", "k", "delta_phi_vh"]].values)
    df_resultado["aceitavel"] = predicoes
    df_filtrado = df_resultado[df_resultado["aceitavel"] == 1].reset_index(drop=True)
    print(f"Amostras geradas: {len(df_resultado)} | Amostras aceitáveis: {len(df_filtrado)}")
except Exception as e:
    print(f"Erro ao classificar: {e}")
    df_filtrado = df_resultado

df_filtrado.to_csv(f"parametros_gerados_setor{SETOR_DESEJADO}.csv", index=False)
