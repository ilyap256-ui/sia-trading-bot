import os
from pathlib import Path

# Основная структура проекта
project_structure = {
    "config": ["__init__.py", "settings.py"],
    "data": ["__init__.py", "data_collector.py"],
    "analysis": ["__init__.py", "levels.py", "patterns.py", "trend.py", "volume.py"],
    "ml": ["__init__.py", "model.py", "features.py"],
    "signals": ["__init__.py", "signal_generator.py", "signal_filter.py"],
    "bot": ["__init__.py", "telegram_bot.py"],
    "utils": ["__init__.py", "logger.py", "scheduler.py"],
}

# Содержимое файлов
file_contents = {
    "config/settings.py": """import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    CURRENCY_PAIRS = ["EUR/USD", "GBP/JPY", "AUD/USD", "EUR/AUD", "EUR/CHF", "AUD/CHF"]
    TIMEFRAME = "1h"
    HISTORICAL_CANDLES = 70
    SESSIONS = {
        "Asian": {"start": 2, "end": 8},
        "European": {"start": 8, "end": 16},
        "American": {"start": 16, "end": 23}
    }

settings = Settings()
""",

    "data/data_collector.py": """import requests
from datetime import datetime, timedelta
from config.settings import settings

class DataCollector:
    def __init__(self):
        self.api_key = settings.TWELVE_DATA_API_KEY
        self.base_url = "https://api.twelvedata.com"

    def get_historical_data(self, symbol, interval, output_size):
        endpoint = f"{self.base_url}/time_series"
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": output_size,
            "apikey": self.api_key
        }
        response = requests.get(endpoint, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to fetch data: {response.text}")

    def get_current_data(self, symbol):
        endpoint = f"{self.base_url}/price"
        params = {
            "symbol": symbol,
            "apikey": self.api_key
        }
        response = requests.get(endpoint, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to fetch data: {response.text}")
""",

    "analysis/levels.py": """import pandas as pd
import numpy as np

class LevelAnalyzer:
    def __init__(self, data):
        self.data = data

    def find_local_levels(self, window=20):
        local_max = self.data['high'].rolling(window, center=True).max()
        local_min = self.data['low'].rolling(window, center=True).min()
        return local_max, local_min

    def find_global_levels(self, window=70):
        global_max = self.data['high'].rolling(window).max()
        global_min = self.data['low'].rolling(window).min()
        return global_max, global_min

    def get_significant_levels(self, local_max, local_min, global_max, global_min):
        significant_levels = []
        for level in [local_max, local_min, global_max, global_min]:
            significant_levels.extend(level.dropna().unique())
        return sorted(set(significant_levels))
""",

    "analysis/patterns.py": """class PatternRecognizer:
    def __init__(self, data):
        self.data = data

    def is_hammer(self, index):
        candle = self.data.iloc[index]
        body = abs(candle['close'] - candle['open'])
        lower_wick = candle['low'] - min(candle['open'], candle['close'])
        upper_wick = max(candle['open'], candle['close']) - candle['high']
        return (body < 0.3 * (candle['high'] - candle['low'])) and (lower_wick > 2 * body) and (upper_wick < 0.1 * body)

    def is_engulfing(self, index):
        prev_candle = self.data.iloc[index - 1]
        curr_candle = self.data.iloc[index]
        return (prev_candle['close'] < prev_candle['open'] and curr_candle['close'] > curr_candle['open'] and
                curr_candle['close'] > prev_candle['open'] and curr_candle['open'] < prev_candle['close'])

    def recognize_patterns(self):
        patterns = []
        for i in range(1, len(self.data)):
            if self.is_hammer(i):
                patterns.append({"index": i, "pattern": "Hammer", "type": "bullish"})
            elif self.is_engulfing(i):
                patterns.append({"index": i, "pattern": "Engulfing", "type": "bullish"})
        return patterns
""",

    "ml/model.py": """import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SignalModel:
    def __init__(self):
        self.model = XGBClassifier()

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        print(f"Model accuracy: {accuracy_score(y_test, y_pred)}")

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]
""",

    "signals/signal_generator.py": """class SignalGenerator:
    def __init__(self, data, levels, patterns, model):
        self.data = data
        self.levels = levels
        self.patterns = patterns
        self.model = model

    def generate_signals(self):
        signals = []
        for pattern in self.patterns:
            signal = {
                "pair": self.data.iloc[pattern["index"]].name,
                "pattern": pattern["pattern"],
                "type": pattern["type"],
                "level": self._get_nearest_level(pattern["index"]),
                "probability": self.model.predict(self._prepare_features(pattern["index"]))
            }
            signals.append(signal)
        return signals

    def _get_nearest_level(self, index):
        price = self.data.iloc[index]['close']
        return min(self.levels, key=lambda x: abs(x - price))

    def _prepare_features(self, index):
        # Подготовка признаков для модели
        pass
""",

    "bot/telegram_bot.py": """from telegram import Bot
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from config.settings import settings

class TelegramBot:
    def __init__(self):
        self.bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
        self.updater = Updater(token=settings.TELEGRAM_BOT_TOKEN, use_context=True)
        self.setup_handlers()

    def setup_handlers(self):
        dp = self.updater.dispatcher
        dp.add_handler(CommandHandler("start", self.start))
        dp.add_handler(MessageHandler(Filters.text, self.send_signal))

    def start(self, update, context):
        update.message.reply_text("🚀 Привет! Я SIA Signals 1.0. Ожидайте торговые сигналы.")

    def send_signal(self, update, context, signal):
        message = self._format_signal(signal)
        self.bot.send_message(chat_id=settings.TELEGRAM_CHAT_ID, text=message)

    def _format_signal(self, signal):
        return f"📈 Новый сигнал: {signal['pair']} - {signal['type']}"

    def run(self):
        self.updater.start_polling()
        self.updater.idle()
""",

    "main.py": """from data.data_collector import DataCollector
from analysis.levels import LevelAnalyzer
from analysis.patterns import PatternRecognizer
from ml.model import SignalModel
from signals.signal_generator import SignalGenerator
from bot.telegram_bot import TelegramBot
from config.settings import settings
import pandas as pd

def main():
    # 1. Сбор данных
    collector = DataCollector()
    historical_data = collector.get_historical_data("EUR/USD", "1h", 70)
    df = pd.DataFrame(historical_data['values'], columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])

    # 2. Анализ уровней
    analyzer = LevelAnalyzer(df)
    local_max, local_min = analyzer.find_local_levels()
    global_max, global_min = analyzer.find_global_levels()
    levels = analyzer.get_significant_levels(local_max, local_min, global_max, global_min)

    # 3. Распознавание паттернов
    recognizer = PatternRecognizer(df)
    patterns = recognizer.recognize_patterns()

    # 4. Машинное обучение
    model = SignalModel()
    # Здесь нужно подготовить данные для обучения
    # model.train(X_train, y_train)

    # 5. Генерация сигналов
    generator = SignalGenerator(df, levels, patterns, model)
    signals = generator.generate_signals()

    # 6. Отправка сигналов в Telegram
    bot = TelegramBot()
    for signal in signals:
        bot.send_signal(None, None, signal)

if __name__ == "__main__":
    main()
""",

    ".env": """TWELVE_DATA_API_KEY=ваш_ключ_от_twelve_data
TELEGRAM_BOT_TOKEN=ваш_токен_телеграм_бота
TELEGRAM_CHAT_ID=ваш_чат_айди
""",

    "requirements.txt": """python-telegram-bot
requests
pandas
numpy
scikit-learn
xgboost
matplotlib
python-dotenv
"""
}

def create_project_structure():
    for directory, files in project_structure.items():
        os.makedirs(directory, exist_ok=True)
        for file in files:
            filepath = os.path.join(directory, file)
            if not os.path.exists(filepath):
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write("# Этот файл создан автоматически\n")

    for filepath, content in file_contents.items():
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

if __name__ == "__main__":
    create_project_structure()
    print("Структура проекта и файлы успешно созданы!")
