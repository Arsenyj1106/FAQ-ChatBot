from config import dp, bot
import handlers.start
import handlers.faq

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())