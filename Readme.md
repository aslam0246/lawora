# ⚖️ **Lawora: AI-Powered Legal Assistant**

[//]: # (Badges removed for private branding)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://opensource.org/license/apache-2-0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1yrS2Kp-kprYWot_sEu7JeWMIRAei_vov?usp=sharing)
[![Loom](https://img.shields.io/badge/Loom-Tutorial-8A2BE2?logo=loom)](https://www.loom.com/share/dcc6b14c653c4618829f46a9aa2ab68c?sid=00d0d3c1-9d4b-4cf7-8684-cdee76718bd5)
[![LangChain](https://img.shields.io/badge/LangChain-Open%20Source-5e9cff?logo=langchain&logoColor=white)](https://python.langchain.com/docs/introduction/)
[![Crew AI](https://img.shields.io/badge/Crew%20AI-Multi--Agent%20Workflows-00bda?style=flat-square)](https://www.crewai.com/) 

### *Bridging the Gap Between People and Legal Access*  🌍

[//]: # (External website link removed for private branding)

**Lawora** is an AI-powered assistant designed to make legal guidance accessible. Using **Retriever-Augmented Generation (RAG)**, **Lexora** delivers quick, contextual legal support.

> 🛡️ **Mission:** “Make legal knowledge accessible, clear, and actionable.”

This project is developed with support from mentors and experts at [Data Science Academy](https://datascience.one/) and [Curvelogics](https://www.curvelogics.com/). 💼

---

## 📚 **Legal Coverage**

Lawora currently supports the following laws, with plans to expand:

- 🏛️ **The Indian Constitution**
- 📜 **The Bharatiya Nyaya Sanhita, 2023**
- 🚨 **The Bharatiya Nagarik Suraksha Sanhita, 2023**
- 🧾 **The Bharatiya Sakshya Adhiniyam, 2023**
- 📦 **The Consumer Protection Act, 2019**
- 🧭 **The Motor Vehicles Act, 1988**
- 💻 **Information Technology Act, 2000**
- 👧 **The Protection of Children from Sexual Offences Act (POCSO), 2012**
- **The Sexual Harassment of Women at Workplace (Prevention, Prohibition and Redressal) Act, 2013**



<div>
    <a href="https://www.loom.com/embed/dcc6b14c653c4618829f46a9aa2ab68c?sid=a5a73b89-88a5-4bc2-a633-f97792f6441f">
      <p>Lexora - Tutorial </p>
    </a>
    <a href=https://www.loom.com/embed/dcc6b14c653c4618829f46a9aa2ab68c?sid=a5a73b89-88a5-4bc2-a633-f97792f6441f">
      <img style="max-width:300px;" src="https://cdn.loom.com/sessions/thumbnails/576b26dcd5fb4d74a3a9e1f8187851bc-35587db59696dfef-full-play.gif">
    </a>
  </div>





---

## 💻 **Developer Quick Start Guide**
---   

Ready to get started? Follow these simple steps to set up **Lexora** on your machine:

1. **Clone the Repository** 🌀
    ```bash
    # Clone your private repository or copy the project folder
    ```

2. **Install uv** 📂

    First, let’s install uv and set up our Python project and environment
    
    MacOS/Linux:
      ``` bash 
      curl -LsSf https://astral.sh/uv/install.sh | sh
      ```

    Windows:

      ``` bash 
      powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
      ```
    Make sure to restart your terminal afterwards to ensure that the uv command gets picked up.

3. **Install Dependencies** 📦
    ```bash
    uv sync
    ```

4. **Set Your Gemini API Key** 🔑

   Open `.env` and add your Gemini API key:
      ```bash
      GEMINI_API_KEY=your-gemini-api-key-here
      ```
      
   **Getting your Gemini API Key:**
   1. Visit [Google AI Studio](https://aistudio.google.com/)
   2. Sign in with your Google account
   3. Navigate to API settings and create a new API key
   4. Copy the generated API key and add it to your `.env` file

5. **Run the Application** 🚀
    ```bash
    uv run streamlit run app.py
    ```

6. **Access the App** 🌐  
    Open your browser and visit:  
    ```bash
    http://127.0.0.1:8501
    ```
---

## 🗄️ **Enable Redis Caching (Recommended for Production Use)**

Lexora can use Redis to cache chat history and LLM responses for faster performance.

### **How to Install and Activate Redis**

1. **Install Redis Server**

   **Ubuntu/Linux:**
   ```bash
    sudo apt-get update
    sudo apt-get install redis-server
   ```
    **MacOS (with Homebrew):**
    ```bash
    brew install redis
    ```
    **Windows (Recommended: Use WSL - Windows Subsystem for Linux):**
   
    1. [Install WSL](https://learn.microsoft.com/en-us/windows/wsl/install) and set up a Linux distribution (e.g., Ubuntu).
    2. Inside the WSL terminal, run:
       ```bash
       sudo apt-get update
       sudo apt-get install redis-server
       ```
    3. Start the Redis server:
       ```bash
       redis-server
       ```

3. **Start Redis Server**
    ```bash
    redis-server
    ```
4. **Verify Redis is Running**
    ```bash
    redis-cli ping
    ```
    You should see: ```PONG```
5. **No Additional Python Setup Needed**  
  - The backend connects to Redis at `redis://localhost:6379/0` if enabled.
  - If you want to use a different host or port, update the `redis_url` parameter in your code.
   Redis caching is optional for local development but **highly recommended** for production deployments to ensure fast and reliable chat experiences.
---

## 🔧 **Tools & Technologies**

| 💡 **Technology**  | 🔍 **Description**                            |
|--------------------|-----------------------------------------------|
| **LangChain**       | Framework for building language models       |
| **ChromaDB**        | Vector database for RAG implementation       |
| **Django**          | High-level Python web framework for robust apps|
| **Gemini API**      | Powering natural language understanding      |

---

## 🌟 **Future Roadmap**

Planned enhancements for **Lawora**:

1.  **🤝 Smarter Together: Introducing Our Multi-Agentic Framework 🤖**
    * Imagine a team of specialized AI agents working seamlessly in the background to provide you with the most comprehensive and efficient legal insights. Our new multi-agent framework makes this a reality, boosting platform performance like never before!

2.  **🌎 Law Without Borders: Expanding Our Global Reach 🇨🇦 + More!**
    * Expanding the legal knowledge base to additional jurisdictions.

3.  **🗣️ Your Voice is the Key: Introducing Voice Interaction 🎙️**
    * Navigate and access legal information effortlessly with our new voice command feature. Simply speak your queries and let Lexora do the rest – making legal research more intuitive and accessible.

4.  **🌍 Bridging Language Barriers: Multi-Lingual Legal Assistance 🌐**
    * Multilingual assistance.

5.  **🎯 Precision & Personalization: Advanced Search & Tailored Assistance 🔍**
    * Say goodbye to endless scrolling! Our enhanced search engine will pinpoint the exact legal information you need with lightning speed. Plus, enjoy personalized suggestions and assistance crafted just for you.

6.  **✍️ Draft with Confidence: Introducing Legal Document Generation 📄**
    * Need a contract or agreement? Our upcoming legal document generation feature will empower you to create essential legal documents using customizable templates and intuitive user input.

7.  **🗓️ Stay Organized, Stay Ahead: Introducing Case Management 📁**
    * Effortlessly manage your legal matters with our new case management feature. Track crucial deadlines, appointments, and important events all in one centralized location, keeping you in control.

---

## 🤝 **Contribute**

[//]: # (Contribution section removed for private deployment)

---

**Lawora** aims to democratize access to legal knowledge.
