import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"


def limpeza_inteligente(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    colunas_remover = ["Cabin", "Ticket", "Name"]
    colunas_existentes = [col for col in colunas_remover if col in df.columns]
    df.drop(columns=colunas_existentes, inplace=True)

    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())

    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    if "SibSp" in df.columns and "Parch" in df.columns:
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    if "Age" in df.columns:
        df["IsChild"] = (df["Age"] < 12).astype(int)

    return df


def main():
    print("Carregando dataset...")
    df = pd.read_csv(DATA_URL)

    print("Aplicando limpeza inteligente...")
    df_ml = limpeza_inteligente(df)

    X = df_ml.drop(columns="Survived")
    y = df_ml["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    colunas_numericas = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    colunas_categoricas = X.select_dtypes(include=["object"]).columns.tolist()

    transformador_numerico = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    transformador_categorico = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessador = ColumnTransformer(transformers=[
        ("num", transformador_numerico, colunas_numericas),
        ("cat", transformador_categorico, colunas_categoricas)
    ])

    modelo_final = Pipeline(steps=[
        ("preprocessador", preprocessador),
        ("modelo", RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        ))
    ])

    print("Treinando modelo...")
    modelo_final.fit(X_train, y_train)

    y_pred = modelo_final.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Acurácia no teste: {acc:.4f}")

    joblib.dump(modelo_final, "modelo_titanic_tunado.pkl")
    print("Modelo salvo com sucesso como modelo_titanic_tunado.pkl")


if __name__ == "__main__":
    main()