from sentence_transformers import SentenceTransformer, util
import numpy as np
import re

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

faq = {
    "Как связаться с техподдержкой?": "Напишите на support@example.com или позвоните по номеру +7 (XXX) XXX-XX-XX.",
    "Где найти контакты поддержки?": "Контакты службы поддержки: email support@example.com, телефон +7 (XXX) XXX-XX-XX.",
    "Куда обращаться за помощью?": "Обратитесь в поддержку: support@example.com.",
}

questions = list(faq.keys())
question_embeddings = model.encode(questions, convert_to_tensor=True)

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # Удаляем пунктуацию
    return text

def get_answer(user_question, threshold=0.4):
    user_question = normalize_text(user_question)
    user_embedding = model.encode(user_question, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(user_embedding, question_embeddings)[0]
    
    best_match_idx = np.argmax(similarities).item()
    best_score = similarities[best_match_idx].item()
    
    if best_score >= threshold:
        return faq[questions[best_match_idx]]
    else:
        return f"Извините, я не нашел ответа (лучший вариант: '{questions[best_match_idx]}' сходство={best_score:.2f})."

# Тест
print(get_answer("поддержка"))