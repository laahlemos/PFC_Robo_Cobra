import pandas as pd
import joblib
from sqlalchemy import create_engine


def extrair_parametros(param_string):
    param_dict = {}
    try:
        for item in param_string.split(", "):
            key, value = item.split(": ")
            param_dict[key.strip()] = float(value)
    except Exception as e:
        print(f"Erro ao processar parâmetros: {param_string}. Erro: {e}")
    return param_dict

engine = create_engine("mysql+mysqlconnector://lara:xxx@127.0.0.1/simulation_db")
query = "SELECT * FROM base_resultados WHERE id IN (1,11,29,31,41,51,63,71,81,91,102,111,121,131,142,152,161,171,181,191,201,211,227,231,241,251,262,272,286,301,302,314,322,332,342,354,366,372,383,396,402,413,422,432,442,452,462,472,483,496,511,512,522,533,542,552,562,572,582,592,602,612,622,632,642,652,662,672,682,692,702,712,722,732,742,752,762,772,789,801,802,812,822,832,843,858,863,873,882,892,902,917,927,937,947,957,971,978,987,1001,1009,1026,1027,1046,1048,1065,1071,1079,1092,1097,1108,1117,1127,1137,1147,1157,1167,1177,1187,1197,1207,1217,1228,1237,1251,1257,1267,1277,1287,1298,1307,1317,1327,1337,1347,1357,1367,1377,1387,1398,1407,1417,1427,1437,1447,1457,1469,1477,1488,1497,1508,1517,1528,1537,1547,1557,1567,1577,1589,1597,1607,1618,1627,1637,1647,1657,1667,1677,1687,1702,1707,1717,1727,1737,1747,1757,1767,1777,1787,1797,1807,1817,1827,1837,1847,1857,1867,1877,1887,1897,1907,1917,1927,1937,1951,1963,1967,1977,1987,2005,2007,2024,2028,2041,2047,2059,2069,2078,2091,2097,2108,2120,2132,2141,2147,2159,2168,2180,2191,2201,2207,2217,2227,2237,2247,2257,2276,2277,2287,2299,2315,2322,2337,2342,2354,2361,2371,2381,2400,2401,2413,2421,2431,2441,2451,2461,2471,2481,2491,2501,2512,2521,2531,2541,2551,2561,2571,2581,2591,2601,2611,2621,2631,2641,2651,2661,2671,2681,2691,2701,2711,2721,2731,2741,2751,2761,2771,2781,2791,2801,2812,2821,2831,2841,2851,2861,2871,2881,2891,2901,2911,2921,2931,2941,2951,2961,2971,2981,2991,3001,3020,3030,3032,3042,3052)"
dados = pd.read_sql(query, engine)

parametros_expandido = dados["parameters"].apply(lambda x: extrair_parametros(x)).apply(pd.Series)
df_completo = pd.concat([dados, parametros_expandido], axis=1)

modelo = joblib.load('modelo_mlp.pkl')
scaler = joblib.load('scaler.pkl')
X = df_completo[['alpha_i', 'alpha_h', 'k', 'delta_phi_vh']]
X_normalizado = scaler.transform(X)

df_completo['result_teste'] = modelo.predict(X_normalizado)
df_completo['result_teste'] = df_completo['result_teste'].apply(lambda x: 'aceitável' if x == 1 else 'não aceitável')

df_completo[['id', 'result', 'result_teste']].to_csv('resultado_com_teste.csv', index=False)
