from aiogram import types
from aiogram.filters import Command
from config import dp

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(
        "Добро пожаловать в университетский FAQ-бот!\n\n"
        "Просто напиши свой вопрос, и я постараюсь помочь!\n"
    )