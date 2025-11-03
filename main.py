import pandas as pd
import numpy as np
import asyncio
from data.data_collector import DataCollector
from analysis.levels import LevelAnalyzer
from analysis.patterns import PatternRecognizer
from ml.model import SignalModel
from signals.signal_generator import SignalGenerator
from bot.telegram_bot import TelegramBot
from stats.statistics_manager import StatisticsManager
from config.settings import settings

async def send_signals(bot, signals):
    for signal in signals:
        await bot.send_signal(signal)

async def main():
    try:
        # 1. Сбор данных
        collector = DataCollector()
        all_pairs_data = collector.get_all_pairs_data("1h", settings.HISTORICAL_CANDLES)

        for pair, data in all_pairs_data.items():
            try:
                df = pd.DataFrame(data['values'], columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
                df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric, errors='coerce')

                print(f"DataFrame for {pair}:")
                print(df.head())

                # 2. Анализ уровней
                analyzer = LevelAnalyzer(df)
                local_max, local_min = analyzer.find_local_levels()
                global_max, global_min = analyzer.find_global_levels()
                levels = analyzer.get_significant_levels(local_max, local_min, global_max, global_min)

                # 3. Распознавание паттернов
                recognizer = PatternRecognizer(df)
                patterns = recognizer.recognize_patterns()

                print(f"Recognized Patterns for {pair}:")
                print(patterns)

                # 4. Машинное обучение
                model = SignalModel()
                X_train, y_train = model.prepare_training_data(df)
                if len(X_train) > 0 and len(y_train) > 0:
                    model.train(X_train, y_train)
                else:
                    print(f"No training data for {pair}")

                # 5. Генерация сигналов
                generator = SignalGenerator(df, levels, patterns, model)
                signals = generator.generate_signals()

                print(f"Generated Signals for {pair}:")
                print(signals)

                # 6. Создание бота и отправка сигналов
                bot = TelegramBot()
                await send_signals(bot, signals)

                # 7. Статистика и аналитика
                stats_manager = StatisticsManager()
                for signal in signals:
                    # Симуляция закрытия сделки
                    exit_price_index = signal['index'] + 1
                    if exit_price_index < len(df):
                        exit_price = df.iloc[exit_price_index]['close']
                    else:
                        exit_price = df.iloc[signal['index']]['close']

                    profit = exit_price - df.iloc[signal['index']]['close'] if signal['type'] == 'bullish' else df.iloc[signal['index']]['close'] - exit_price

                    trade = {
                        'pair': pair,
                        'pattern': signal['pattern'],
                        'type': signal['type'],
                        'level': signal['level'],
                        'probability': signal['probability'],
                        'volume': signal['volume'],
                        'entry_price': df.iloc[signal['index']]['close'],
                        'exit_price': exit_price,
                        'profit': profit
                    }
                    stats_manager.add_trade(trade)

                stats_report = stats_manager.generate_report()
                print(stats_report)
                await bot.bot.send_message(chat_id=int(settings.TELEGRAM_CHAT_ID), text=stats_report)

            except Exception as e:
                print(f"Error processing {pair}: {e}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
