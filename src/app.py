import os
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI


if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "key"


llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

df_header = pd.read_csv("../docs/202401_NFs_Cabecalho.csv")
df_itens = pd.read_csv("../docs/202401_NFs_Itens.csv")

df_completo = pd.merge(df_itens, df_header, on="CHAVE DE ACESSO", how="left")


agent = create_pandas_dataframe_agent(
    llm, 
    df_completo,
    verbose=True, 
    allow_dangerous_code=True,
    agent_executor_kwargs={
        "system_message" : """
            Você é um contador, especialista em notas ficais eletronicas,
            analisa, conta  e organiza essas notas para exibir para o usuario,
            o usuario vai fazer perguntas sobre as notas fiscais e voce deve focar inteiramente nisso,
            quando necessário explique os calculos e utilize termos contabeis e fiscais quando fizer sentido,
            sempre que o usuario perguntar sobre algum item e não especificar, use a descrição, responde em portugues as perguntas
        """
    })


pergunta = input("Faça uma pergunta: ")
resposta = agent.run(pergunta)

print("\nResposta do Agente:\n", resposta)
