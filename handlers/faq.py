from aiogram import types
from config import dp
from ai_faq import FAQGenerator

# Инициализируем генератор один раз при старте
faq_gen = FAQGenerator()

@dp.message()
async def handle_question(message: types.Message):
    answer = faq_gen.get_answer(message.text)
    await message.answer(answer)