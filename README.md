# CustomGPT-Med: AI-Powered Mining of Medical & Scientific Publications 🚀

This project demonstrates how a custom fine-tuned GPT-based model can be used for mining scientific publications, clinical trial reports, and patents. It enables tasks like semantic search, named entity recognition (NER), automatic summarization, and question-answering for medical publications, pharmaceutical innovation, and healthcare research.

---

## 🔍 Key Features

1. **Data Collection from PubMed**: Extract abstracts and full-text articles from PubMed using Python scripts and APIs.
2. **Entity Recognition**: Extracts biomedical entities such as diseases, drugs, genes, and experimental methods.
3. **Fine-Tuning GPT**: Fine-tunes GPT for scientific and medical domain language.
4. **Semantic Search**: Perform semantic search on medical publications using Sentence Transformers.
5. **Question-Answering**: Implements a question-answering pipeline for answering specific questions based on research data.
6. **Summarization**: Provides summarization of lengthy research papers or clinical trial reports.
7. **Patent Interpretation**: Analyzes patents for innovation tracking in pharmaceutical research.

---

## 📂 Project Structure

1. **`goal_1_t1_data_collection.py`**:
   - Collect data from medical publication sources like PubMed and ClinicalTrials.gov using APIs.
   
2. **`goal_1_t2_data_preprocessing.py`**:
   - Preprocess the text data by cleaning and tokenizing for GPT fine-tuning.
   
3. **`goal_1_t3_gpt_fine_tuning.py`**:
   - Fine-tune the GPT-2 model on biomedical data to enhance domain-specific language understanding.
   
4. **`goal_2_entity_extraction.py`**:
   - Named Entity Recognition (NER) to extract biomedical terms such as drugs, genes, and diseases using Hugging Face Transformers.

5. **`goal_2_viz.py`**:
   - Visualizes the extracted entities using word clouds and frequency charts.
   
6. **`goal_3_t1_semantic_search.py`**:
   - Perform semantic search to find the most relevant scientific papers or trials based on a user query.

7. **`goal_3_t2_question_answering.py`**:
   - Implement a question-answering pipeline using pre-trained models to answer specific questions from research data.

8. **`goal_3_t3_app.py`**:
   - Create a web app interface using Gradio that allows users to input search queries and questions.

9. **`goal_4_summarization.py`**:
   - Summarizes lengthy clinical trial reports or research papers into concise summaries.

10. **`goal_5_patent_interpretation.py`**:
    - Analyzes patents from USPTO and other sources to track innovation in pharmaceutical research.

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/SaiDhanyaa/CustomGPT-Med-AI-Powered-Mining-of-Medical-Scientific-Publications
cd CustomGPT-Med-AI-Powered-Mining-of-Medical-Scientific-Publications
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Fine-Tune the GPT Model
To fine-tune the GPT model on the collected medical data:
```bash
python goal_1_t3_gpt_fine_tuning.py
```

### 4. Named Entity Recognition (NER)
To perform NER on the collected data:
```bash
python goal_2_entity_extraction.py
```

### 5. Visualize Entities
To visualize extracted entities:
```bash
python goal_2_viz.py
```

### 6. Semantic Search
To run a semantic search:
```bash
python goal_3_t1_semantic_search.py
```

### 7. Question-Answering
To run the question-answering system:
```bash
python goal_3_t2_question_answering.py
```

### 8. Summarization
To summarize a long clinical trial or research paper:
```bash
python goal_4_summarization.py
```

### 9. Patent Interpretation
To analyze and interpret patents:
```bash
python goal_5_patent_interpretation.py
```

### 10. Run the Web App
To launch the Gradio-based web app:
```bash
python goal_3_t3_app.py
```

---

## 🎯 Project Goals

1. **Objective 1**: Collect and preprocess data from sources like PubMed and ClinicalTrials.gov.
2. **Objective 2**: Fine-tune GPT on scientific/medical publications for better understanding.
3. **Objective 3**: Implement entity extraction to identify key terms like diseases, drugs, and methods.
4. **Objective 4**: Implement a search and question-answering pipeline to assist researchers.
5. **Objective 5**: Provide patent analysis to track pharmaceutical innovations.

---

## 📈 Example Outputs

- **Entity Extraction**: Frequency charts and word clouds showing the most common biomedical entities.
- **Semantic Search**: Retrieves the top 5 abstracts for a given query.
- **Question-Answering**: Answers specific questions based on research data.
- **Summarization**: Concise summaries of long research documents.
- **Patent Interpretation**: Insights into pharmaceutical research patents.

---

## 🔧 Future Improvements

- **Expand NER**: Add support for extracting more specialized entities (e.g., clinical symptoms).
- **Advanced Summarization**: Implement more advanced summarization techniques.
- **Real-Time Data**: Streamline real-time data from PubMed for immediate access to publications.
- **Patent Search**: Improve the patent interpretation module for better pharmaceutical innovation tracking.

---

## 📝 License

This project is licensed under the MIT License. See the LICENSE file for more details.

---

## 👨‍💻 Author

- **Dhanyapriya Somasundaram** ([@SaiDhanyaa](https://github.com/SaiDhanyaa))

For any queries or collaborations, feel free to contact me via [email](mailto:dhanyapriyas@arizona.edu).
