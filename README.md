# **Sistema RAG com LLM Local**

Um sistema completo de **Geração Aumentada por Recuperação (RAG - Retrieval-Augmented Generation)** que opera de forma totalmente local. A aplicação permite fazer perguntas em linguagem natural e receber respostas baseadas em um conjunto de documentos privados, garantindo privacidade e controle total dos dados.

## **Índice**

  - [1. Visão Geral](https://www.google.com/search?q=%231-vis%C3%A3o-geral)
  - [2. Funcionalidades Principais](https://www.google.com/search?q=%232-funcionalidades-principais)
  - [3. Arquitetura e Componentes](https://www.google.com/search?q=%233-arquitetura-e-componentes)
  - [4. Pré-requisitos](https://www.google.com/search?q=%234-pr%C3%A9-requisitos)
  - [5. Guia de Instalação e Configuração](https://www.google.com/search?q=%235-guia-de-instala%C3%A7%C3%A3o-e-configura%C3%A7%C3%A3o)
  - [6. Como Usar](https://www.google.com/search?q=%236-como-usar)
  - [7. Endpoints da API](https://www.google.com/search?q=%237-endpoints-da-api)
  - [8. Estrutura do Código](https://www.google.com/search?q=%238-estrutura-do-c%C3%B3digo)
  - [9. Solução de Problemas (FAQ)](https://www.google.com/search?q=%239-solu%C3%A7%C3%A3o-de-problemas-faq)
  - [10. Licença](https://www.google.com/search?q=%2310-licen%C3%A7a)

## **1. Visão Geral**

O objetivo deste projeto é fornecer uma solução RAG ponta-a-ponta que seja fácil de configurar e usar localmente. O sistema indexa documentos de vários formatos, armazena-os em um banco de dados vetorial e utiliza um LLM local para gerar respostas contextualmente relevantes para as perguntas dos usuários.

## **2. Funcionalidades Principais**

  - **Processamento de Múltiplos Formatos:** Extrai texto de:
      - `PDF` (texto nativo e imagens via OCR)
      - `DOCX` (Microsoft Word)
      - `XLSX` (Microsoft Excel)
      - `TXT` (Texto puro)
      - Imagens (`JPG`, `PNG`) via OCR.
  - **Indexação Inteligente:** Converte o conteúdo dos documentos em embeddings usando `SentenceTransformers` e os armazena no `ChromaDB` para busca semântica rápida.
  - **Servidor de IA Local:** Gerencia automaticamente um servidor para o LLM (`llama-server` do `llama.cpp`), garantindo que a capacidade de geração de texto esteja sempre disponível.
  - **API Robusta:** Oferece endpoints via `Flask` para interagir com o sistema, verificar status e gerenciar a indexação.
  - **Auditoria:** Registra todas as interações (perguntas, respostas, contexto) em um banco de dados `SQLite` para rastreabilidade.
  - **Processamento em Background:** Novos arquivos enviados são indexados em uma thread separada para não bloquear a resposta ao usuário.

## **3. Arquitetura e Componentes**

  - **Backend (API Flask):** Ponto de entrada que orquestra a comunicação entre os componentes.
  - **Processador de Documentos:** Funções que usam `PyMuPDF`, `docx2python`, `openpyxl`, `Pillow`, `OpenCV` e `Tesseract OCR` para extração de texto.
  - **Modelo de Embedding (`SentenceTransformer`):** Converte texto em vetores numéricos para busca semântica.
  - **Banco de Dados Vetorial (`ChromaDB`):** Armazena os vetores e metadados para recuperação rápida de contexto.
  - **Servidor do LLM (`llama-server`):** Executa um modelo de linguagem (formato GGUF) em um processo separado para gerar as respostas.
  - **Banco de Dados de Auditoria (`SQLite`):** Armazena o histórico de interações.

## **4. Pré-requisitos**

Antes de começar, garanta que você tenha os seguintes softwares instalados:

  - **Python 3.9+** e **Pip**.
  - **Tesseract-OCR:**
      - Necessário para OCR em imagens e PDFs escaneados.
      - Instale a partir do [repositório oficial do Tesseract](https://github.com/tesseract-ocr/tesseract).
      - **Importante:** Adicione o Tesseract ao PATH do seu sistema ou configure o caminho no script.
  - **Llama.cpp:**
      - O script requer o comando `llama-server`. A melhor forma de obtê-lo é compilando o projeto a partir do [repositório oficial do Llama.cpp](https://github.com/ggerganov/llama.cpp).
      - Após a compilação, certifique-se de que o executável `llama-server` esteja acessível pelo PATH do sistema.
  - **Modelo de Linguagem (LLM):**
      - Um modelo de linguagem no formato **GGUF**.
      - Você pode baixar modelos do Hugging Face (ex: [modelos quantizados da TheBloke](https://huggingface.co/TheBloke)).

## **5. Guia de Instalação e Configuração**

### **Passo 1: Preparar o Ambiente**

1.  Clone este repositório ou salve o código como `rag_app.py`.

2.  Crie e ative um ambiente virtual:

    ```bash
    # Crie o ambiente virtual
    python -m venv venv

    # Ative no Windows
    .\venv\Scripts\activate

    # Ative no Linux/macOS
    source venv/bin/activate
    ```

### **Passo 2: Instalar as Dependências**

1.  Crie um arquivo `requirements.txt` com o seguinte conteúdo:

    ```text
    Flask
    Flask-Cors
    werkzeug
    PyMuPDF
    docx2python
    Pillow
    pytesseract
    opencv-python-headless
    numpy
    requests
    sentence-transformers
    langchain-text-splitters
    chromadb
    torch
    openpyxl
    llama-cpp-python
    ```

2.  Instale todas as bibliotecas:

    ```bash
    pip install -r requirements.txt
    ```

    > **Nota para GPU (NVIDIA):** Para garantir que o PyTorch utilize sua GPU, pode ser necessário um comando de instalação específico. Consulte o [site oficial do PyTorch](https://pytorch.org/get-started/locally/) para obter o comando correto para sua versão do CUDA.

### **Passo 3: Configurar os Caminhos no Script**

Abra o arquivo `rag_app.py` e edite as variáveis de configuração no topo do arquivo para corresponder ao seu ambiente.

  - `DOCUMENTS_DIR`: Crie e defina o caminho para a pasta que conterá seus documentos.
  - `TESSERACT_CMD_PATH`: Se o Tesseract não estiver no PATH, defina o caminho completo para `tesseract.exe`.
  - `LLAMA_MODEL_PATH`: **(Crítico)** Defina o caminho completo para o arquivo do modelo `.gguf` que você baixou.

### **Passo 4: Adicionar Documentos**

Copie seus arquivos (`.pdf`, `.docx`, `.txt`, etc.) para o diretório especificado em `DOCUMENTS_DIR`.

## **6. Como Usar**

### **Modo 1: Indexação Inicial (Obrigatório)**

Antes de executar a aplicação, você deve indexar seus documentos.

```bash
python rag_app.py indexar
```

Este comando irá ler todos os arquivos, extrair o texto e popular o banco de dados vetorial. Este processo só precisa ser executado na primeira vez ou quando desejar uma reindexação completa.

### **Modo 2: Executar o Servidor**

Após a indexação, inicie a aplicação.

```bash
python rag_app.py
```

O script irá iniciar o `llama-server` em segundo plano e, em seguida, o servidor Flask, que ficará disponível em `http://localhost:5000`.

## **7. Endpoints da API**

### `GET /status`

Verifica a saúde e o estado da aplicação.

  - **URL:** `http://localhost:5000/status`
  - **Resposta de Sucesso (200):**
    ```json
    {
        "servidor_flask_online": true,
        "servidor_llm_online": true,
        "diretorio_documentos_configurado": "C:\\Documentos",
        "diretorio_documentos_existe": true,
        "total_chunks_indexados_chromadb": 150,
        "colecao_chromadb_vazia": false,
        "modelo_embedding_carregado": true,
        "tesseract_configurado": true
    }
    ```

### `POST /reindexar`

Dispara uma reindexação completa dos documentos, apagando o índice antigo e criando um novo.

  - **URL:** `http://localhost:5000/reindexar`
  - **Corpo:** Nenhum
  - **Resposta de Sucesso (200):**
    ```json
    {
        "mensagem": "Reindexação completa dos documentos concluída com sucesso!"
    }
    ```

### `POST /perguntar`

Endpoint principal para fazer perguntas.

  - **URL:** `http://localhost:5000/perguntar`
  - **Corpo:** `multipart/form-data`
  - **Parâmetros do Formulário:**

| Parâmetro          | Tipo   | Obrigatório | Descrição                                                        |
| :----------------- | :----- | :---------- | :--------------------------------------------------------------- |
| `pergunta`         | string | Sim         | A pergunta do usuário.                                           |
| `arquivo`          | file   | Não         | Um arquivo para ser usado como contexto adicional para a pergunta. |
| `incluir_contexto` | string | Não         | Se `'true'`, inclui o contexto exato enviado ao LLM na resposta.  |

  - **Resposta de Sucesso (200):**
    ```json
    {
        "pergunta": "Qual o valor do projeto X?",
        "resposta": "Com base no arquivo 'proposta_final.docx', o valor total do projeto X é de R$ 50.000,00. [Fonte(s) utilizada(s): proposta_final.docx]",
        "arquivos_consultados": [
            "proposta_final.docx",
            "relatorio_custos.xlsx"
        ],
        "contexto_utilizado": null
    }
    ```

## **8. Estrutura do Código**

  - **SEÇÃO: CONFIGURAÇÃO DE LOGGING:** Define o log para console e arquivo.
  - **SEÇÃO: CAMINHOS E PARÂMETROS GLOBAIS:** Centraliza todas as constantes e configurações.
  - **SEÇÃO: PARÂMETROS DO SERVIDOR LLAMA:** Parâmetros específicos para a execução do `llama-server`.
  - **SEÇÃO: GERENCIAMENTO DO SERVIDOR LLAMA:** Funções para iniciar, parar e verificar o processo do LLM.
  - **SEÇÃO: BANCO DE DADOS (CHROMA DB E SQLITE):** Funções de inicialização e interação com os bancos de dados.
  - **SEÇÃO: PROCESSAMENTO E EXTRAÇÃO DE TEXTO:** Lógica de extração de texto para cada tipo de arquivo suportado, incluindo OCR.
  - **SEÇÃO: INDEXAÇÃO E GERAÇÃO DE CHUNKS:** Orquestração do processo de indexação, desde a leitura dos arquivos até a inserção no ChromaDB.
  - **SEÇÃO: RECUPERAÇÃO DE CONTEXTO E GERAÇÃO DE RESPOSTA (RAG):** O núcleo da lógica RAG, combinando recuperação de contexto e geração de texto.
  - **SEÇÃO: ENDPOINTS DA API FLASK:** Definição das rotas da API.
  - **SEÇÃO: BLOCO PRINCIPAL DE EXECUÇÃO:** Ponto de entrada que diferencia a execução entre modo `indexar` e modo `servidor`.

## **9. Solução de Problemas (FAQ)**

  - **Erro `ModuleNotFoundError`:**
      - Certifique-se de que seu ambiente virtual está ativado e que você executou `pip install -r requirements.txt`.
  - **Servidor Llama não inicia:**
      - Verifique se o `LLAMA_MODEL_PATH` está correto e aponta para um arquivo `.gguf` válido.
      - Execute o comando `llama-server` manualmente no seu terminal para ver mensagens de erro detalhadas.
      - Se estiver usando GPU (`NGL > 0`), verifique se os drivers da NVIDIA e o CUDA estão instalados corretamente.
  - **Erro `Tesseract not found`:**
      - Verifique se o Tesseract está instalado e se o seu caminho de instalação foi adicionado ao PATH do sistema.
      - Caso contrário, defina o caminho explícito na variável `TESSERACT_CMD_PATH` no script.
  - **As respostas são genéricas ou "não encontrei a informação":**
      - Execute a reindexação (`python rag_app.py indexar`) para garantir que seus documentos estão no banco de dados.
      - Verifique os logs (`rag_app.log`) para ver se ocorreram erros durante a extração de texto dos seus arquivos.
      - Tente ajustar os parâmetros `CHUNK_SIZE` e `CHUNK_OVERLAP` e reindexar.

## **10. Licença**

Este projeto é distribuído sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.
