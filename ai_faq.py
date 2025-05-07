from sentence_transformers import SentenceTransformer, CrossEncoder, util
import numpy as np
import json
from pathlib import Path

class FAQGenerator:
    def __init__(self):
        # Загрузка моделей
        self.bi_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.cross_model = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
        
        # Загрузка датасета
        dataset_path = Path(__file__).parent / 'dataset.json'
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.faq = {item['question']: item['answer'] for item in json.load(f)}
        self.questions = list(self.faq.keys())
        
        # Предварительное кодирование вопросов
        self.question_embeddings = self.bi_model.encode(self.questions, convert_to_tensor=True)

    def get_answer(self, user_question):
        # Этап 1: Быстрый поиск кандидатов
        user_embedding = self.bi_model.encode(user_question, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(user_embedding, self.question_embeddings)[0]
        candidate_indices = np.where(similarities > 0.6)[0]
        
        if len(candidate_indices) == 0:
            return "Извините, не нашел подходящего ответа в базе знаний."
        
        # Этап 2: Точное ранжирование кандидатов
        candidates = [(user_question, self.questions[i]) for i in candidate_indices]
        scores = self.cross_model.predict(candidates)
        best_idx = candidate_indices[np.argmax(scores)]
        
        return self.faq[self.questions[best_idx]]