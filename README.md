# GenAI Practice Example: Talk to a Database

This is an end-to-end LLM project utilizing Google Palm and Langchain. We are developing a system that interfaces with a MySQL database. Users ask questions in natural language, and the system responds by converting those questions into SQL queries and executing them on the MySQL database. XYZ, a T-shirt store, manages its inventory, sales, and discount data in a MySQL database. A store manager might ask questions such as:

  1. How many white Adidas T-shirts do we have left in stock?
  2. What will our store's sales be if we sell all extra-small T-shirts after applying discounts?

   ![T Shirt Store GUI for User](https://github.com/priyadse/GenAI_Practice/assets/68457424/a2fdf437-891a-4e42-b8ee-bd5e2f4c4ff3)

# Project Highlights

XYZ is a T-shirt store that offers Adidas, Nike, Van Heusen, and Levi's T-shirts. Their inventory, sales, and discount data are stored in a MySQL database. We will develop an LLM-based question-and-answer system using the following components:

  1. Google Palm LLM
  2. Hugging Face embeddings
  3. Streamlit for the user interface
  4. Langchain framework
  5. ChromaDB as a vector store
  6. Few-shot learning
In the UI, the store manager will ask questions in natural language, and the system will generate the corresponding answers.

# Installation

1. Clone this repository to your local machine using:
   
```bash
  git clone https://github.com/priyadse/GenAI_Practice.git
```

2. Install the required dependencies using pip: (switch version based on your local system requirements and adjust dependencies accordingly)

```bash
  pip install -r requirements.txt
```

3. Acquire an API key from https://aistudio.google.com/app/ and add it to the .env file:

```bash
    GOOGLE_API_KEY="your_api_key_here"
```

4. For database setup, run db_creation_t_shirts.sql in your MySQL Workbench.

# Usage

1. Run the Streamlit app by executing:

```bash
  streamlit run main.py
```

2. The web app will open in your browser where you can ask questions.

# Sample Questions

    1. How many total T-shirts are left in stock?
    2. How many Nike T-shirts do we have left in XS size and white color?
    3. What is the total price of the inventory for all S-size T-shirts?
    4. How much sales amount will be generated if we sell all small-size Adidas shirts today after discounts?

# Project Structure

    main.py: The main Streamlit application script.
    langchain_helper.py: Contains all the Langchain code.
    requirements.txt: A list of required Python packages for the project.
    few_shots.py: Contains few-shot prompts.
    .env: Configuration file for storing your Google API key.
   
