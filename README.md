# Titanic Survival Streamlit

Projeto de machine learning com interface em Streamlit para prever a probabilidade de sobrevivência de um passageiro do Titanic.

A ideia aqui foi montar um projeto pequeno, mas completo: carregar dados, tratar variáveis, treinar um modelo e disponibilizar uma interface simples para testar previsões.

# O que o projeto faz

O usuário informa dados do passageiro, como:

- classe
- sexo
- idade
- número de familiares a bordo
- tarifa paga
- porto de embarque

Com base nisso, o modelo estima a probabilidade de sobrevivência.

# Como o projeto foi organizado

Separei o projeto em dois arquivos principais:

- `train_model.py`: faz o carregamento dos dados, a limpeza, o treinamento e salva o modelo
- `app.py`: carrega o modelo treinado e exibe a interface no Streamlit

Essa separação foi importante porque, no começo, tentei usar um modelo salvo em outro ambiente e tive erro de compatibilidade entre versões do `scikit-learn`. Gerar o modelo localmente resolveu o problema e deixou o projeto mais confiável.

 Tecnologias usadas

- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib

Como rodar localmente

1. Instale as dependências:

```bash
pip install -r requirements.txt

2. Treine o modelo localmente:

```bash
python train_model.py

3. Rode a aplicação:

```bash
streamlit run app.py

4. Abra no navegador:

http://localhost:8501

## Desafios encontrados:

Durante o desenvolvimento, os principais pontos foram:

incompatibilidade ao carregar um modelo .pkl treinado em outro ambiente
necessidade de reduzir o custo do treino para rodar bem em uma máquina com 8 GB de RAM
separação entre pipeline de treino e app para evitar retrabalho e erros futuros

O que eu pratiquei nesse projeto

Nesse projeto eu treinei principalmente:

limpeza e preparação de dados
criação de pipeline de machine learning
treinamento de modelo com Scikit-learn
construção de interface interativa com Streamlit
organização de projeto para GitHub

Próximos passos

Algumas melhorias que ainda quero fazer:

publicar o app no Streamlit Cloud
melhorar o visual da interface
salvar histórico de previsões
testar outros modelos além do Random Forest


Autor

Rômulo M. Corrêa



