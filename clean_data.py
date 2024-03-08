"""Taller evaluable presencial"""

import nltk
import pandas as pd
from pandas import DataFrame

nltk.download("punkt")


def load_data(input_file: str) -> DataFrame:
    """Lea el archivo usando pandas y devuelva un DataFrame"""
    dataframe: DataFrame = pd.read_csv(input_file, sep="\t", names=["text"])
    return dataframe


def create_fingerprint(df: DataFrame) -> DataFrame:
    """Cree una nueva columna en el DataFrame que contenga el fingerprint de la columna 'text'"""
    dataframe: DataFrame = df.copy()
    # 1. Copie la columna 'text' a la columna 'fingerprint'
    dataframe["fingerprint"] = dataframe["text"]
    # 2. Remueva los espacios en blanco al principio y al final de la cadena
    dataframe["fingerprint"] = dataframe["fingerprint"].str.strip()
    # 3. Convierta el texto a minúsculas
    dataframe["fingerprint"] = dataframe["fingerprint"].str.lower()
    # 4. Transforme palabras que pueden (o no) contener guiones por su version sin guion.
    # 5. Remueva puntuación y caracteres de control
    dataframe["fingerprint"] = dataframe["fingerprint"].str.replace(
        "[^A-Za-z0-9\s]", "", regex=True
    )
    # 6. Convierta el texto a una lista de tokens
    dataframe["fingerprint"] = dataframe["fingerprint"].apply(
        lambda x: " ".join(nltk.word_tokenize(x))
    )
    # 7. Transforme cada palabra con un stemmer de Porter
    dataframe["fingerprint"] = dataframe["fingerprint"].apply(
        lambda x: " ".join([nltk.PorterStemmer().stem(word) for word in x.split()])
    )
    # 8. Ordene la lista de tokens y remueve duplicados
    # 9. Convierta la lista de tokens a una cadena de texto separada por espacios
    dataframe["fingerprint"] = dataframe["fingerprint"].apply(
        lambda x: " ".join(sorted(set(x.split())))
    )

    return dataframe


def generate_cleaned_column(df: DataFrame) -> DataFrame:
    """Crea la columna 'cleaned' en el DataFrame"""

    df = df.copy()

    # 1. Ordene el dataframe por 'fingerprint' y 'text'
    df = df.sort_values(by=["fingerprint", "text"])
    # 2. Seleccione la primera fila de cada grupo de 'fingerprint'
    df2: DataFrame = df.drop_duplicates(subset=["fingerprint"], keep="first")
    # 3.  Cree un diccionario con 'fingerprint' como clave y 'text' como valor
    dictionary: dict[str, str] = df2.set_index("fingerprint").to_dict()["text"]
    # 4. Cree la columna 'cleaned' usando el diccionario
    df["cleaned"] = df["fingerprint"].map(dictionary)
    return df


def save_data(df: DataFrame, output_file: str) -> None:
    """Guarda el DataFrame en un archivo"""
    # Solo contiene una columna llamada 'texto' al igual
    # que en el archivo original pero con los datos limpios
    df["text"] = df["cleaned"]
    df.to_csv(output_file, sep="\t", columns=["text"], index=False, header=True)


def main(input_file: str, output_file: str):
    """Ejecuta la limpieza de datos"""

    df: DataFrame = load_data(input_file)
    df = create_fingerprint(df)
    df = generate_cleaned_column(df)
    df.to_csv("test.csv", index=False)
    save_data(df, output_file)


if __name__ == "__main__":
    main(
        input_file="input.txt",
        output_file="output.txt",
    )
