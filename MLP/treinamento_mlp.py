import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import joblib

def extrair_parametros(param_string):
    param_dict = {}
    try:
        for item in param_string.split(", "):
            key, value = item.split(": ")
            param_dict[key.strip()] = float(value)
    except Exception as e:
        print(f"Erro ao processar parâmetros: {param_string}. Erro: {e}")
    return param_dict


engine = create_engine("mysql+mysqlconnector://lara:xxx!@127.0.0.1/simulation_db")
query = "SELECT * FROM base_resultados"
dados = pd.read_sql(query, engine)

parametros_expandido = dados["parameters"].apply(lambda x: extrair_parametros(x)).apply(pd.Series)
df_completo = pd.concat([dados, parametros_expandido], axis=1)
X = df_completo[['alpha_i', 'alpha_h', 'k', 'delta_phi_vh']]
y = df_completo['result'].apply(lambda x: 1 if x == 'aceitável' else 0)

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, alpha=0.1, random_state=42, learning_rate_init=0.001)
mlp.fit(X_train, y_train)

score = mlp.score(X_test, y_test)
print(f"Acurácia do modelo: {score:.2f}")


joblib.dump(mlp, 'modelo_mlp.pkl')
joblib.dump(scaler, 'scaler.pkl')
