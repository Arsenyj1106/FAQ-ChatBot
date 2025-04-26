from sentence_transformers import SentenceTransformer, CrossEncoder, util
import numpy as np
import json

# Загрузка моделей (используем существующие модели)
bi_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # Для поиска кандидатов
cross_model = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')  # Рабочая мультиязычная модель

# Загрузка датасета
with open('dataset.json', 'r', encoding='utf-8') as f:
    faq = {item['question']: item['answer'] for item in json.load(f)}
questions = list(faq.keys())

# Предварительное кодирование вопросов
question_embeddings = bi_model.encode(questions, convert_to_tensor=True)

def get_answer(user_question):
    # Этап 1: Быстрый поиск кандидатов
    user_embedding = bi_model.encode(user_question, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(user_embedding, question_embeddings)[0]
    candidate_indices = np.where(similarities > 0.4)[0]  # Берем все с похожестью >40%
    
    if len(candidate_indices) == 0:
        return "Извините, не нашел подходящего ответа."
    
    # Этап 2: Точное ранжирование кандидатов
    candidates = [(user_question, questions[i]) for i in candidate_indices]
    scores = cross_model.predict(candidates)
    best_idx = candidate_indices[np.argmax(scores)]
    
    return faq[questions[best_idx]]

# Тестирование
test_questions = [
    "как подать документы",
    "что нужно для поступления",
    "проходные баллы"
]

for q in test_questions:
    print(f"Вопрос: {q}")
    print(f"Ответ: {get_answer(q)}\n")