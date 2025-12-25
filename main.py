import logging
import os
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, ContextTypes, filters
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
CLASSES = ["дельфин", "кит", "тюлень"]
MODEL = keras.models.load_model("sea_animals_model1.h5")
logging.basicConfig(level=logging.INFO)
def predict_sea_animal(image_path, img_size=256):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((img_size, img_size))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = MODEL.predict(img)
    class_index = np.argmax(prediction)
    return CLASSES[class_index]

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Отправь мне фото морского животного, и я скажу, кто это!")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    file = await photo.get_file()
    file_path = f"temp_{update.message.chat.id}.jpg"
    await file.download_to_drive(file_path)

    try:
        result = predict_sea_animal(file_path)
        await update.message.reply_text(f"Вероятнее всего это: {result}")
    except Exception as e:
        await update.message.reply_text("Произошла ошибка при обработке изображения.")
        logging.error(e)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
if __name__ == '__main__':
    TOKEN = "токен типа да"
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    print("Бот запущен...")
    app.run_polling()
