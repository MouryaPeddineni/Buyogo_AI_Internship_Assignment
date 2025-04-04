import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
from typing import Dict, Any, List, Optional
from pathlib import Path

# For RAG components
import faiss
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import torch

# For API (unused in this version, but kept for future expansion)
from fastapi import FastAPI
import uvicorn
import sqlite3

# Initialize project structure
PROJECT_ROOT = Path(file).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DB_PATH = DATA_DIR / "hotel_bookings.db"

# Ensure directories exist
for dir_path in [DATA_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# DataProcessor and AnalyticsEngine classes remain unchanged
class DataProcessor:
    def init(self, data_path: Optional[str] = None):
        self.data = None
        self.processed_data = None
        self.db_connection = None
        if data_path:
            self.load_data(data_path)

    def load_data(self, data_path: str) -> bool:
        try:
            self.data = pd.read_csv(data_path).sample(1000, random_state=42)
            print(f"Data loaded successfully: {self.data.shape[0]} records with {self.data.shape[1]} features.")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    # def preprocess_data(self) -> pd.DataFrame:
    #     if self.data is None:
    #         print("No data loaded. Please load data first.")
    #         return None
    #     self.processed_data = self.data.copy()
    #     # Exclude both arrival_date_month and arrival_date_year from datetime conversion
    #     date_columns = [col for col in self.processed_data.columns if
    #                     'date' in col.lower() and col not in ['arrival_date_month', 'arrival_date_year']]
    #     for col in date_columns:
    #         try:
    #             self.processed_data[col] = pd.to_datetime(self.processed_data[col], errors='coerce')
    #         except Exception as e:
    #             print(f"Could not convert column {col} to datetime: {e}")
    #     num_columns = self.processed_data.select_dtypes(include=['float64', 'int64']).columns
    #     for col in num_columns:
    #         self.processed_data[col] = self.processed_data[col].fillna(self.processed_data[col].median())
    #     cat_columns = self.processed_data.select_dtypes(include=['object']).columns
    #     for col in cat_columns:
    #         self.processed_data[col] = self.processed_data[col].fillna(self.processed_data[col].mode()[0])
    #     if 'reservation_status_date' in self.processed_data.columns and 'arrival_date' in self.processed_data.columns:
    #         self.processed_data['lead_time_days'] = (self.processed_data['arrival_date'] -
    #                                                  self.processed_data['reservation_status_date']).dt.days.abs()
    #     if 'arrival_date_year' in self.processed_data.columns and 'arrival_date_month' in self.processed_data.columns and 'arrival_date_day_of_month' in self.processed_data.columns:
    #         self.processed_data['arrival_date'] = pd.to_datetime(
    #             self.processed_data['arrival_date_year'].astype(str) + '-' +
    #             self.processed_data['arrival_date_month'] + '-' +
    #             self.processed_data['arrival_date_day_of_month'].astype(str),
    #             format='%Y-%B-%d',
    #             errors='coerce'
    #         )
    #     if 'adr' in self.processed_data.columns and 'stays_in_weekend_nights' in self.processed_data.columns and 'stays_in_week_nights' in self.processed_data.columns:
    #         self.processed_data['total_nights'] = self.processed_data['stays_in_weekend_nights'] + self.processed_data[
    #             'stays_in_week_nights']
    #         self.processed_data['total_price'] = self.processed_data['adr'] * self.processed_data['total_nights']
    #     print("Data preprocessing completed.")
    #     return self.processed_data

    def preprocess_data(self) -> pd.DataFrame:
        if self.data is None:
            print("No data loaded. Please load data first.")
            return None
        self.processed_data = self.data.copy()

        # Ensure arrival_date is created first
        if all(col in self.processed_data.columns for col in
               ['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']):
            self.processed_data['arrival_date'] = pd.to_datetime(
                self.processed_data['arrival_date_year'].astype(str) + '-' +
                self.processed_data['arrival_date_month'] + '-' +
                self.processed_data['arrival_date_day_of_month'].astype(str),
                format='%Y-%B-%d',
                errors='coerce'
            )
        else:
            print("Warning: Could not create arrival_date; required columns missing.")

        # Convert reservation_status_date to datetime
        if 'reservation_status_date' in self.processed_data.columns:
            self.processed_data['reservation_status_date'] = pd.to_datetime(
                self.processed_data['reservation_status_date'], errors='coerce'
            )

        # Calculate lead_time_days
        if 'reservation_status_date' in self.processed_data.columns and 'arrival_date' in self.processed_data.columns:
            self.processed_data['lead_time_days'] = (
                    self.processed_data['arrival_date'] - self.processed_data['reservation_status_date']
            ).dt.days.abs()
            print(f"Lead time calculated: {self.processed_data['lead_time_days'].describe()}")
        else:
            print("Warning: Could not calculate lead_time_days; required columns missing.")

        # Rest of the preprocessing (numeric and categorical filling, total_nights, total_price)
        num_columns = self.processed_data.select_dtypes(include=['float64', 'int64']).columns
        for col in num_columns:
            self.processed_data[col] = self.processed_data[col].fillna(self.processed_data[col].median())
        cat_columns = self.processed_data.select_dtypes(include=['object']).columns
        for col in cat_columns:
            self.processed_data[col] = self.processed_data[col].fillna(self.processed_data[col].mode()[0])

        if 'adr' in self.processed_data.columns and 'stays_in_weekend_nights' in self.processed_data.columns and 'stays_in_week_nights' in self.processed_data.columns:
            self.processed_data['total_nights'] = self.processed_data['stays_in_weekend_nights'] + self.processed_data[
                'stays_in_week_nights']
            self.processed_data['total_price'] = self.processed_data['adr'] * self.processed_data['total_nights']

        print("Data preprocessing completed.")
        return self.processed_data

    def initialize_database(self) -> None:
        try:
            self.db_connection = sqlite3.connect(str(DB_PATH))
            cursor = self.db_connection.cursor()
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS bookings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hotel_type TEXT,
                lead_time INTEGER,
                arrival_date DATE,
                stays_in_weekend_nights INTEGER,
                stays_in_week_nights INTEGER,
                adults INTEGER,
                children INTEGER,
                meal TEXT,
                country TEXT,
                market_segment TEXT,
                distribution_channel TEXT,
                is_repeated_guest INTEGER,
                previous_cancellations INTEGER,
                previous_bookings_not_canceled INTEGER,
                reserved_room_type TEXT,
                assigned_room_type TEXT,
                booking_changes INTEGER,
                deposit_type TEXT,
                days_in_waiting_list INTEGER,
                customer_type TEXT,
                adr REAL,
                required_car_parking_spaces INTEGER,
                total_of_special_requests INTEGER,
                reservation_status TEXT,
                reservation_status_date DATE,
                total_price REAL,
                is_canceled INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS query_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                response TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            self.db_connection.commit()
            print("Database initialized successfully.")
        except Exception as e:
            print(f"Error initializing database: {e}")

    def store_data_in_db(self) -> bool:
        if self.processed_data is None:
            print("No processed data available. Please preprocess data first.")
            return False
        if self.db_connection is None:
            self.initialize_database()
        try:
            df_to_store = self.processed_data.copy()
            column_mapping = {'hotel': 'hotel_type'}
            for old_col, new_col in column_mapping.items():
                if old_col in df_to_store.columns:
                    df_to_store.rename(columns={old_col: new_col}, inplace=True)
            df_to_store.to_sql('bookings', self.db_connection, if_exists='replace', index=False)
            print(f"Successfully stored {len(df_to_store)} records in the database.")
            return True
        except Exception as e:
            print(f"Error storing data in database: {e}")
            return False


class AnalyticsEngine:
    def init(self, data_processor: DataProcessor):
        self.data_processor = data_processor
        self.db_connection = data_processor.db_connection

    def get_revenue_trends(self, time_period: str = 'monthly') -> Dict[str, Any]:
        if self.data_processor.processed_data is None:
            return {"error": "No data available for analysis."}
        df = self.data_processor.processed_data
        if 'arrival_date' not in df.columns or 'total_price' not in df.columns:
            return {"error": "Required columns for revenue analysis not found."}
        if 'is_canceled' in df.columns:
            df = df[df['is_canceled'] == 0]
        if time_period == 'monthly':
            df['period'] = df['arrival_date'].dt.to_period('M').astype(str)
        elif time_period == 'quarterly':
            df['period'] = df['arrival_date'].dt.to_period('Q').astype(str)
        elif time_period == 'yearly':
            df['period'] = df['arrival_date'].dt.to_period('Y').astype(str)
        else:
            return {"error": "Invalid time period. Choose 'monthly', 'quarterly', or 'yearly'."}
        revenue_by_period = df.groupby('period')['total_price'].sum().reset_index()
        revenue_by_period.columns = ['period', 'revenue']
        return {
            "time_period": time_period,
            "revenue_data": revenue_by_period.to_dict(orient='records'),
            "total_revenue": float(revenue_by_period['revenue'].sum())
        }

    def get_lead_time_distribution(self) -> Dict[str, Any]:
        """Calculate and return booking lead time distribution statistics."""
        if self.data_processor.processed_data is None:
            return {"error": "No data available for analysis."}
        df = self.data_processor.processed_data
        if 'lead_time_days' not in df.columns:
            return {"error": "Lead time data not available in the dataset."}

        lead_time_stats = df['lead_time_days'].describe().to_dict()
        lead_time_bins = pd.cut(df['lead_time_days'], bins=[0, 7, 30, 90, 180, float('inf')],
                                labels=['0-7 days', '8-30 days', '31-90 days', '91-180 days', '180+ days'])
        lead_time_distribution = (lead_time_bins.value_counts() / len(df) * 100).round(2).to_dict()

        # Optional: Generate a histogram (commented out for CLI execution, enable for visualization)
        # plt.hist(df['lead_time_days'], bins=30)
        # plt.title("Booking Lead Time Distribution")
        # plt.xlabel("Lead Time (days)")
        # plt.ylabel("Frequency")
        # plt.savefig("lead_time_distribution.png")

        return {
            "found": True,
            "lead_time_stats": {
                "mean": round(lead_time_stats['mean'], 2),
                "median": round(lead_time_stats['50%'], 2),
                "min": int(lead_time_stats['min']),
                "max": int(lead_time_stats['max'])
            },
            "lead_time_distribution": lead_time_distribution
        }

from transformers import pipeline
# RAGEngine class as previously extended (assuming it's in the same file or imported)
class RAGEngine:
    def init(self, data_processor: DataProcessor):
        self.data_processor = data_processor
        self.db_connection = data_processor.db_connection
        self.embedding_model = None
        self.index = None
        self.text_data = []
        self.analytics_engine = AnalyticsEngine(data_processor)
        self.booking_data = data_processor.processed_data
        self.total_rooms = 120
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        self.llm = pipeline("text-generation", model="EleutherAI/gpt-neo-125M")

    def initialize_embedding_model(self, model_name: str = 'all-MiniLM-L6-v2') -> None:
        try:
            self.embedding_model = SentenceTransformer(model_name)
            print(f"Embedding model '{model_name}' loaded successfully.")
        except Exception as e:
            print(f"Error loading embedding model: {e}")

    def prepare_text_data(self) -> None:
        if self.booking_data is None:
            print("No processed data available. Please preprocess data first.")
            return
        df = self.booking_data
        text_data = []
        for i, row in df.iterrows():
            doc = {'id': i, 'text': '', 'metadata': {}}
            text_parts = []
            if 'hotel' in row:
                text_parts.append(f"Booking at {row['hotel']} hotel.")
                doc['metadata']['hotel_type'] = row['hotel']
            if 'arrival_date' in row:
                arrival_date = row['arrival_date']
                text_parts.append(f"Arrival date: {arrival_date}.")
                doc['metadata']['arrival_date'] = str(arrival_date)
                if isinstance(arrival_date, pd.Timestamp):
                    doc['metadata']['arrival_month'] = arrival_date.month_name()
                    doc['metadata']['arrival_year'] = arrival_date.year
                    text_parts.append(f"Arrival in {arrival_date.month_name()} {arrival_date.year}.")
            if 'country' in row:
                text_parts.append(f"Guest from {row['country']}.")
                doc['metadata']['country'] = row['country']
            if 'adr' in row:
                text_parts.append(f"Average daily rate: {row['adr']}.")
                doc['metadata']['adr'] = float(row['adr'])
            if 'total_price' in row:
                text_parts.append(f"Total price: {row['total_price']}.")
                doc['metadata']['total_price'] = float(row['total_price'])
            if 'lead_time' in row:
                text_parts.append(f"Booked {row['lead_time']} days in advance.")
                doc['metadata']['lead_time'] = int(row['lead_time'])
            if 'stays_in_weekend_nights' in row and 'stays_in_week_nights' in row:
                total_nights = row['stays_in_weekend_nights'] + row['stays_in_week_nights']
                text_parts.append(f"Stayed for {total_nights} nights total.")
                doc['metadata']['total_nights'] = int(total_nights)
            if 'is_canceled' in row:
                status = "Canceled" if row['is_canceled'] == 1 else "Confirmed"
                text_parts.append(f"Reservation status: {status}.")
                doc['metadata']['is_canceled'] = int(row['is_canceled'])
            if 'total_of_special_requests' in row:
                text_parts.append(f"Made {row['total_of_special_requests']} special requests.")
                doc['metadata']['special_requests'] = int(row['total_of_special_requests'])
            doc['text'] = " ".join(text_parts)
            text_data.append(doc)
        self.text_data = text_data
        print(f"Prepared {len(text_data)} text entries for embedding.")

    def build_faiss_index(self) -> None:
        if not self.text_data:
            print("No text data available. Please prepare text data first.")
            return
        if self.embedding_model is None:
            self.initialize_embedding_model()
        try:
            texts = [doc['text'] for doc in self.text_data]
            print(f"Encoding {len(texts)} text entries with SentenceTransformer...")
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            print("Normalizing embeddings...")
            faiss.normalize_L2(embeddings)
            vector_dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(vector_dimension)
            print("Adding embeddings to FAISS index...")
            self.index.add(embeddings)
            print(f"FAISS index built successfully with {len(texts)} vectors of dimension {vector_dimension}.")
        except Exception as e:
            print(f"Error building FAISS index: {e}")
            raise

    def retrieve_relevant_context(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None:
            print("FAISS index not initialized. Please build index first.")
            return []
        if self.embedding_model is None:
            self.initialize_embedding_model()
        try:
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)
            scores, indices = self.index.search(query_embedding, top_k)
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.text_data):
                    result = {
                        "text": self.text_data[idx]['text'],
                        "metadata": self.text_data[idx]['metadata'],
                        "score": float(scores[0][i])
                    }
                    results.append(result)
            return results
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return []

    def execute_sql_query(self, sql_query: str) -> pd.DataFrame:
        if self.db_connection is None:
            print("Database connection not available.")
            return pd.DataFrame()
        try:
            return pd.read_sql_query(sql_query, self.db_connection)
        except Exception as e:
            print(f"Error executing SQL query: {e}")
            return pd.DataFrame()

    # def handle_analytics_query(self, query: str) -> Dict[str, Any]:
    #     query_lower = query.lower()
    #     import re
    #
    #     # Check for year-specific revenue queries
    #     year_match = re.search(r'(\d{4})', query_lower)
    #     if year_match and any(term in query_lower for term in ['revenue', 'sales', 'income', 'earnings']):
    #         year = int(year_match.group(1))
    #         for month in ['january', 'february', 'march', 'april', 'may', 'june',
    #                       'july', 'august', 'september', 'october', 'november', 'december']:
    #             if month in query_lower:
    #                 return self.get_monthly_revenue_by_year(month, year)
    #
    #     # Existing logic for other cases
    #     if any(term in query_lower for term in ['revenue', 'sales', 'income', 'earnings']):
    #         if any(month in query_lower for month in ['january', 'february', 'march', 'april', 'may', 'june',
    #                                                   'july', 'august', 'september', 'october', 'november',
    #                                                   'december']):
    #             for month in ['january', 'february', 'march', 'april', 'may', 'june',
    #                           'july', 'august', 'september', 'october', 'november', 'december']:
    #                 if month in query_lower:
    #                     return self.get_monthly_revenue(month)
    #         elif 'year' in query_lower or 'annual' in query_lower:
    #             return self.get_annual_revenue()
    #         elif 'quarter' in query_lower or 'quarterly' in query_lower:
    #             quarter_match = re.search(r'q(\d)', query_lower)
    #             if quarter_match:
    #                 quarter = int(quarter_match.group(1))
    #                 return self.get_quarterly_revenue(quarter)
    #             else:
    #                 return self.get_all_quarterly_revenue()
    #         else:
    #             return self.get_total_revenue()
    #     elif any(term in query_lower for term in ['occupancy', 'booking', 'filled', 'vacancy', 'vacant']):
    #         if 'rate' in query_lower or 'percentage' in query_lower:
    #             if any(month in query_lower for month in ['january', 'february', 'march', 'april', 'may', 'june',
    #                                                       'july', 'august', 'september', 'october', 'november',
    #                                                       'december']):
    #                 for month in ['january', 'february', 'march', 'april', 'may', 'june',
    #                               'july', 'august', 'september', 'october', 'november', 'december']:
    #                     if month in query_lower:
    #                         return self.get_monthly_occupancy_rate(month)
    #             else:
    #                 return self.get_average_occupancy_rate()
    #         else:
    #             return self.get_total_bookings()
    #     elif any(term in query_lower for term in ['customer', 'guest', 'client', 'visitor']):
    #         if 'repeat' in query_lower or 'returning' in query_lower:
    #             return self.get_repeat_customer_stats()
    #         elif 'satisfaction' in query_lower or 'rating' in query_lower or 'review' in query_lower:
    #             return self.get_customer_satisfaction()
    #         elif 'demographic' in query_lower or 'nationality' in query_lower or 'country' in query_lower:
    #             return self.get_customer_demographics()
    #         else:
    #             return self.get_general_customer_stats()
    #     elif any(
    #             term in query_lower for term in ['season', 'seasonal', 'summer', 'winter', 'spring', 'fall', 'autumn']):
    #         if 'summer' in query_lower:
    #             return self.get_seasonal_analysis('summer')
    #         elif 'winter' in query_lower:
    #             return self.get_seasonal_analysis('winter')
    #         elif 'spring' in query_lower:
    #             return self.get_seasonal_analysis('spring')
    #         elif 'fall' in query_lower or 'autumn' in query_lower:
    #             return self.get_seasonal_analysis('fall')
    #         else:
    #             return self.get_all_seasonal_analysis()
    #     elif any(term in query_lower for term in ['room type', 'suite', 'deluxe', 'standard']):
    #         if 'popular' in query_lower or 'most booked' in query_lower:
    #             return self.get_popular_room_types()
    #         elif 'revenue' in query_lower:
    #             return self.get_room_type_revenue()
    #         else:
    #             return self.get_room_type_stats()
    #     elif any(term in query_lower for term in ['length of stay', 'duration', 'nights']):
    #         # Check for specific segments in the query
    #         segment = None
    #         if 'business' in query_lower or 'corporate' in query_lower:
    #             segment = 'Corporate'
    #         elif 'online' in query_lower:
    #             segment = 'Online TA'
    #         elif 'group' in query_lower or 'groups' in query_lower:
    #             segment = 'Groups'
    #         return self.get_length_of_stay_stats(segment=segment)
    #     elif 'cancellation' in query_lower and 'pattern' in query_lower:
    #         return self.get_cancellation_patterns()
    #     elif any(term in query_lower for term in ['lead time', 'booking advance', 'days before']):
    #         return self.analytics_engine.get_lead_time_distribution()
    #     elif any(term in query_lower for term in ['channel', 'booking source', 'booking platform', 'online', 'agent']):
    #         return self.get_booking_channel_stats()
    #     elif any(term in query_lower for term in ['insight', 'summary', 'overview', 'analysis']):
    #         return self.get_general_analytics()
    #     context = "\n\n".join([doc['text'] for doc in self.retrieve_relevant_context(query)])
    #     prompt = f"Question: {query}\nContext: {context}\nAnswer in natural language:"
    #     response = self.llm(prompt, max_length=200, num_return_sequences=1)[0]['generated_text']
    #     return {
    #         "found": True,
    #         "response": response.split("Answer in natural language:")[-1].strip(),
    #         "sources_used": [{"type": "booking_data", "relevance_score": 0.9}]
    #     }

    def handle_analytics_query(self, query: str) -> Dict[str, Any]:
        query_lower = query.lower()
        import re
        response = None  # Variable to store rule-based output

        # Check for year-specific revenue queries
        year_match = re.search(r'(\d{4})', query_lower)
        if year_match and any(term in query_lower for term in ['revenue', 'sales', 'income', 'earnings']):
            year = int(year_match.group(1))
            for month in ['january', 'february', 'march', 'april', 'may', 'june',
                          'july', 'august', 'september', 'october', 'november', 'december']:
                if month in query_lower:
                    response = self.get_monthly_revenue_by_year(month, year)
                    break

        # Existing logic for other cases
        if response is None and any(term in query_lower for term in ['revenue', 'sales', 'income', 'earnings']):
            if any(month in query_lower for month in ['january', 'february', 'march', 'april', 'may', 'june',
                                                      'july', 'august', 'september', 'october', 'november',
                                                      'december']):
                for month in ['january', 'february', 'march', 'april', 'may', 'june',
                              'july', 'august', 'september', 'october', 'november', 'december']:
                    if month in query_lower:
                        response = self.get_monthly_revenue(month)
                        break
            elif 'year' in query_lower or 'annual' in query_lower:
                response = self.get_annual_revenue()
            elif 'quarter' in query_lower or 'quarterly' in query_lower:
                quarter_match = re.search(r'q(\d)', query_lower)
                if quarter_match:
                    quarter = int(quarter_match.group(1))
                    response = self.get_quarterly_revenue(quarter)
                else:
                    response = self.get_all_quarterly_revenue()
            else:
                response = self.get_total_revenue()

        elif response is None and any(
                term in query_lower for term in ['occupancy', 'booking', 'filled', 'vacancy', 'vacant']):
            if 'rate' in query_lower or 'percentage' in query_lower:
                if any(month in query_lower for month in ['january', 'february', 'march', 'april', 'may', 'june',
                                                          'july', 'august', 'september', 'october', 'november',
                                                          'december']):
                    for month in ['january', 'february', 'march', 'april', 'may', 'june',
                                  'july', 'august', 'september', 'october', 'november', 'december']:
                        if month in query_lower:
                            response = self.get_monthly_occupancy_rate(month)
                            break
                else:
                    response = self.get_average_occupancy_rate()
            else:
                response = self.get_total_bookings()

        elif response is None and any(term in query_lower for term in ['customer', 'guest', 'client', 'visitor']):
            if 'repeat' in query_lower or 'returning' in query_lower:
                response = self.get_repeat_customer_stats()
            elif 'satisfaction' in query_lower or 'rating' in query_lower or 'review' in query_lower:
                response = self.get_customer_satisfaction()
            elif 'demographic' in query_lower or 'nationality' in query_lower or 'country' in query_lower:
                response = self.get_customer_demographics()
            else:
                response = self.get_general_customer_stats()

        elif response is None and any(
                term in query_lower for term in ['season', 'seasonal', 'summer', 'winter', 'spring', 'fall', 'autumn']):
            if 'summer' in query_lower:
                response = self.get_seasonal_analysis('summer')
            elif 'winter' in query_lower:
                response = self.get_seasonal_analysis('winter')
            elif 'spring' in query_lower:
                response = self.get_seasonal_analysis('spring')
            elif 'fall' in query_lower or 'autumn' in query_lower:
                response = self.get_seasonal_analysis('fall')
            else:
                response = self.get_all_seasonal_analysis()

        elif response is None and any(term in query_lower for term in ['room type', 'suite', 'deluxe', 'standard']):
            if 'popular' in query_lower or 'most booked' in query_lower:
                response = self.get_popular_room_types()
            elif 'revenue' in query_lower:
                response = self.get_room_type_revenue()
            else:
                response = self.get_room_type_stats()

        elif response is None and any(term in query_lower for term in ['length of stay', 'duration', 'nights']):
            segment = None
            if 'business' in query_lower or 'corporate' in query_lower:
                segment = 'Corporate'
            elif 'online' in query_lower:
                segment = 'Online TA'
            elif 'group' in query_lower or 'groups' in query_lower:
                segment = 'Groups'
            response = self.get_length_of_stay_stats(segment=segment)

        elif response is None and 'cancellation' in query_lower and 'pattern' in query_lower:
            response = self.get_cancellation_patterns()

        elif response is None and any(term in query_lower for term in ['lead time', 'booking advance', 'days before']):
            response = self.analytics_engine.get_lead_time_distribution()

        elif response is None and any(
                term in query_lower for term in ['channel', 'booking source', 'booking platform', 'online', 'agent']):
            response = self.get_booking_channel_stats()

        elif response is None and any(term in query_lower for term in ['insight', 'summary', 'overview', 'analysis']):
            response = self.get_general_analytics()

        # If no rule matches, use LLM with RAG directly
        if response is None:
            context = "\n\n".join([doc['text'] for doc in self.retrieve_relevant_context(query)])
            prompt = f"Question: {query}\nContext: {context}\nAnswer in natural language:"
            llm_response = self.llm(prompt, max_length=500, num_return_sequences=1)[0]['generated_text']
            response = {
                "found": True,
                "response": llm_response.split("Answer in natural language:")[-1].strip(),
                "sources_used": [{"type": "booking_data", "relevance_score": 0.9}]
            }

        # Structure the response using LLM
        if response.get("found", False):
            # Convert rule-based response to string for LLM processing
            response_str = str(response)
            context = "\n\n".join([doc['text'] for doc in self.retrieve_relevant_context(query)])
            prompt = (
                f"Question: {query}\n"
                f"Raw Data: {response_str}\n"
                f"Context: {context}\n"
                f"Task: Present the raw data in a readable, natural language format. "
                f"Ensure all key information is included clearly and concisely."
            )
            llm_output = self.llm(prompt, max_length=1000, num_return_sequences=1)[0]['generated_text']
            structured_response = llm_output.split("Task:")[0].strip()  # Extract the generated part
            return {
                "found": True,
                "response": structured_response,
                "sources_used": [{"type": "booking_data", "relevance_score": 0.9}]
            }
        else:
            return {
                "found": False,
                "message": "No relevant data found for the query."
            }

    def get_monthly_revenue(self, month: str) -> Dict[str, Any]:
        month_data = self.booking_data[self.booking_data['arrival_date_month'] == month.title()]
        if month_data.empty:
            return {"found": False, "message": f"No booking data available for {month.title()}."}
        revenue = (month_data['adr'] * month_data['stays_in_weekend_nights'] +
                   month_data['adr'] * month_data['stays_in_week_nights']).sum()
        current_year = self.booking_data['arrival_date_year'].max()
        if isinstance(current_year, pd.Timestamp):
            current_year = current_year.year
        else:
            current_year = int(current_year)
        prev_year_month_data = self.booking_data[
            (self.booking_data['arrival_date_month'] == month.title()) &
            (self.booking_data['arrival_date_year'] == current_year - 1)
            ]
        if not prev_year_month_data.empty:
            prev_revenue = (prev_year_month_data['adr'] * prev_year_month_data['stays_in_weekend_nights'] +
                            prev_year_month_data['adr'] * prev_year_month_data['stays_in_week_nights']).sum()
            yoy_change = ((revenue - prev_revenue) / prev_revenue) * 100
        else:
            yoy_change = None
        return {
            "found": True,
            "month": month.title(),
            "revenue": round(revenue, 2),
            "booking_count": len(month_data),
            "average_daily_rate": round(month_data['adr'].mean(), 2),
            "year_over_year_change": round(yoy_change, 2) if yoy_change is not None else None,
            "currency": "USD"
        }

    def get_annual_revenue(self) -> Dict[str, Any]:
        current_year = max(self.booking_data['arrival_date_year'])
        annual_data = self.booking_data[self.booking_data['arrival_date_year'] == current_year]
        if annual_data.empty:
            return {"found": False, "message": f"No booking data available for year {current_year}."}
        annual_revenue = (annual_data['adr'] * annual_data['stays_in_weekend_nights'] +
                          annual_data['adr'] * annual_data['stays_in_week_nights']).sum()
        monthly_revenue = {}
        for month in annual_data['arrival_date_month'].unique():
            month_data = annual_data[annual_data['arrival_date_month'] == month]
            month_revenue = (month_data['adr'] * month_data['stays_in_weekend_nights'] +
                             month_data['adr'] * month_data['stays_in_week_nights']).sum()
            monthly_revenue[month] = round(month_revenue, 2)
        prev_year_data = self.booking_data[self.booking_data['arrival_date_year'] == current_year - 1]
        if not prev_year_data.empty:
            prev_year_revenue = (prev_year_data['adr'] * prev_year_data['stays_in_weekend_nights'] +
                                 prev_year_data['adr'] * prev_year_data['stays_in_week_nights']).sum()
            yoy_change = ((annual_revenue - prev_year_revenue) / prev_year_revenue) * 100
        else:
            yoy_change = None
        return {
            "found": True,
            "year": current_year,
            "annual_revenue": round(annual_revenue, 2),
            "monthly_breakdown": monthly_revenue,
            "booking_count": len(annual_data),
            "average_daily_rate": round(annual_data['adr'].mean(), 2),
            "year_over_year_change": round(yoy_change, 2) if yoy_change is not None else None,
            "currency": "USD"
        }

    def get_quarterly_revenue(self, quarter: int) -> Dict[str, Any]:
        if quarter not in [1, 2, 3, 4]:
            return {"found": False, "message": "Invalid quarter. Please use 1, 2, 3, or 4."}
        quarter_months = {1: ['January', 'February', 'March'], 2: ['April', 'May', 'June'],
                          3: ['July', 'August', 'September'], 4: ['October', 'November', 'December']}
        current_year = max(self.booking_data['arrival_date_year'])
        quarter_data = self.booking_data[
            (self.booking_data['arrival_date_year'] == current_year) &
            (self.booking_data['arrival_date_month'].isin(quarter_months[quarter]))
            ]
        if quarter_data.empty:
            return {"found": False, "message": f"No booking data available for Q{quarter} {current_year}."}
        revenue = (quarter_data['adr'] * quarter_data['stays_in_weekend_nights'] +
                   quarter_data['adr'] * quarter_data['stays_in_week_nights']).sum()
        return {
            "found": True,
            "quarter": f"Q{quarter} {current_year}",
            "revenue": round(revenue, 2),
            "booking_count": len(quarter_data),
            "average_daily_rate": round(quarter_data['adr'].mean(), 2),
            "currency": "USD"
        }

    def get_all_quarterly_revenue(self) -> Dict[str, Any]:
        current_year = max(self.booking_data['arrival_date_year'])
        quarterly_revenue = {}
        for q in range(1, 5):
            result = self.get_quarterly_revenue(q)
            if result["found"]:
                quarterly_revenue[result["quarter"]] = {
                    "revenue": result["revenue"],
                    "booking_count": result["booking_count"],
                    "average_daily_rate": result["average_daily_rate"]
                }
        return {
            "found": True,
            "year": current_year,
            "quarterly_revenue": quarterly_revenue,
            "currency": "USD"
        }

    def get_total_revenue(self) -> Dict[str, Any]:
        revenue = (self.booking_data['adr'] * self.booking_data['stays_in_weekend_nights'] +
                   self.booking_data['adr'] * self.booking_data['stays_in_week_nights']).sum()
        return {
            "found": True,
            "total_revenue": round(revenue, 2),
            "booking_count": len(self.booking_data),
            "average_daily_rate": round(self.booking_data['adr'].mean(), 2),
            "currency": "USD"
        }

    # def get_average_occupancy_rate(self) -> Dict[str, Any]:
    #     daily_room_nights = self.booking_data.groupby(
    #         ['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']
    #     )['total_nights'].sum()
    #     avg_daily_room_nights = daily_room_nights.mean()
    #     occupancy_rate = (avg_daily_room_nights / self.total_rooms) * 100
    #     monthly_occupancy = {}
    #     month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
    #                    7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
    #     for month in self.booking_data['arrival_date_month'].unique():
    #         month_str = month if isinstance(month, str) else month_names.get(int(month), str(month))
    #         month_data = self.booking_data[self.booking_data['arrival_date_month'] == month]
    #         month_room_nights = month_data.groupby(['arrival_date_year', 'arrival_date_day_of_month'])[
    #             'total_nights'].sum()
    #         month_occupancy_rate = (month_room_nights.mean() / self.total_rooms) * 100
    #         monthly_occupancy[month_str] = round(month_occupancy_rate, 2)
    #     return {
    #         "found": True,
    #         "average_occupancy_rate": round(occupancy_rate, 2),
    #         "monthly_breakdown": monthly_occupancy,
    #         "highest_occupancy_month": max(monthly_occupancy, key=monthly_occupancy.get),
    #         "lowest_occupancy_month": min(monthly_occupancy, key=monthly_occupancy.get)
    #     }

    def get_average_occupancy_rate(self) -> Dict[str, Any]:
        if self.booking_data is None or 'total_nights' not in self.booking_data.columns:
            return {"found": False, "message": "Booking data or total_nights not available."}

        # Calculate total room-nights and days
        total_room_nights = self.booking_data['total_nights'].sum()
        unique_days = len(
            self.booking_data.groupby(['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']))
        avg_daily_room_nights = total_room_nights / unique_days
        occupancy_rate = (avg_daily_room_nights / self.total_rooms) * 100

        # Monthly breakdown
        monthly_occupancy = {}
        month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                       7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
        for month in self.booking_data['arrival_date_month'].unique():
            month_str = month if isinstance(month, str) else month_names.get(int(month), str(month))
            month_data = self.booking_data[self.booking_data['arrival_date_month'] == month]
            month_room_nights = month_data['total_nights'].sum()
            month_days = len(month_data.groupby(['arrival_date_year', 'arrival_date_day_of_month']))
            month_occupancy_rate = (month_room_nights / month_days / self.total_rooms) * 100
            monthly_occupancy[month_str] = round(month_occupancy_rate, 2)

        return {
            "found": True,
            "average_occupancy_rate": round(occupancy_rate, 2),
            "monthly_breakdown": monthly_occupancy,
            "highest_occupancy_month": max(monthly_occupancy, key=monthly_occupancy.get),
            "lowest_occupancy_month": min(monthly_occupancy, key=monthly_occupancy.get),
            "total_rooms_used": self.total_rooms
        }

    def get_monthly_occupancy_rate(self, month: str) -> Dict[str, Any]:
        month_data = self.booking_data[self.booking_data['arrival_date_month'] == month.title()]
        if month_data.empty:
            return {"found": False, "message": f"No booking data available for {month.title()}."}
        daily_bookings = month_data.groupby(['arrival_date_year', 'arrival_date_day_of_month']).size()
        avg_daily_occupancy = daily_bookings.mean()
        occupancy_rate = (avg_daily_occupancy / self.total_rooms) * 100
        return {
            "found": True,
            "month": month.title(),
            "occupancy_rate": round(occupancy_rate, 2),
            "average_daily_bookings": round(avg_daily_occupancy, 2)
        }

    def get_total_bookings(self) -> Dict[str, Any]:
        return {"found": True, "total_bookings": len(self.booking_data)}

    def get_customer_demographics(self) -> Dict[str, Any]:
        country_counts = self.booking_data['country'].value_counts()
        total_bookings = len(self.booking_data)
        country_percentages = (country_counts / total_bookings * 100).round(2)
        top_countries = country_percentages.head(10).to_dict()
        market_segments = (self.booking_data['market_segment'].value_counts() / total_bookings * 100).round(2).to_dict()
        guest_types = {}
        solo_travelers = len(self.booking_data[(self.booking_data['adults'] == 1) &
                                               (self.booking_data['children'] == 0) &
                                               (self.booking_data['babies'] == 0)])
        guest_types['Solo'] = round((solo_travelers / total_bookings) * 100, 2)
        couples = len(self.booking_data[(self.booking_data['adults'] == 2) &
                                        (self.booking_data['children'] == 0) &
                                        (self.booking_data['babies'] == 0)])
        guest_types['Couples'] = round((couples / total_bookings) * 100, 2)
        families = len(self.booking_data[(self.booking_data['children'] > 0) |
                                         (self.booking_data['babies'] > 0)])
        guest_types['Families'] = round((families / total_bookings) * 100, 2)
        groups = len(self.booking_data[self.booking_data['adults'] >= 3])
        guest_types['Groups'] = round((groups / total_bookings) * 100, 2)
        return {
            "found": True,
            "total_bookings": total_bookings,
            "top_countries": top_countries,
            "market_segments": market_segments,
            "guest_types": guest_types
        }

    def get_repeat_customer_stats(self) -> Dict[str, Any]:
        if 'is_repeated_guest' not in self.booking_data.columns:
            return {"found": False, "message": "Repeat guest data not available."}
        total_bookings = len(self.booking_data)
        repeated_guests = self.booking_data['is_repeated_guest'].sum()
        repeat_pct = (repeated_guests / total_bookings) * 100
        return {
            "found": True,
            "total_bookings": total_bookings,
            "repeated_guests": int(repeated_guests),
            "repeat_percentage": round(repeat_pct, 2)
        }

    def get_customer_satisfaction(self) -> Dict[str, Any]:
        return {"found": False, "message": "Customer satisfaction data not available in the current dataset."}

    def get_general_customer_stats(self) -> Dict[str, Any]:
        total_bookings = len(self.booking_data)
        avg_adults = self.booking_data['adults'].mean()
        avg_children = self.booking_data['children'].mean() if 'children' in self.booking_data.columns else 0
        avg_babies = self.booking_data['babies'].mean() if 'babies' in self.booking_data.columns else 0
        return {
            "found": True,
            "total_bookings": total_bookings,
            "average_adults_per_booking": round(avg_adults, 2),
            "average_children_per_booking": round(avg_children, 2),
            "average_babies_per_booking": round(avg_babies, 2)
        }

    def get_seasonal_analysis(self, season: str) -> Dict[str, Any]:
        season_months = {
            'spring': ['March', 'April', 'May'],
            'summer': ['June', 'July', 'August'],
            'fall': ['September', 'October', 'November'],
            'winter': ['December', 'January', 'February']
        }
        if season not in season_months:
            return {"found": False, "message": "Invalid season. Use 'spring', 'summer', 'fall', or 'winter'."}
        season_data = self.booking_data[self.booking_data['arrival_date_month'].isin(season_months[season])]
        if season_data.empty:
            return {"found": False, "message": f"No booking data available for {season}."}
        revenue = (season_data['adr'] * season_data['stays_in_weekend_nights'] +
                   season_data['adr'] * season_data['stays_in_week_nights']).sum()
        return {
            "found": True,
            "season": season,
            "revenue": round(revenue, 2),
            "booking_count": len(season_data),
            "average_daily_rate": round(season_data['adr'].mean(), 2)
        }

    def get_all_seasonal_analysis(self) -> Dict[str, Any]:
        seasonal_data = {}
        for season in ['spring', 'summer', 'fall', 'winter']:
            result = self.get_seasonal_analysis(season)
            if result["found"]:
                seasonal_data[season] = {
                    "revenue": result["revenue"],
                    "booking_count": result["booking_count"],
                    "average_daily_rate": result["average_daily_rate"]
                }
        return {"found": True, "seasonal_analysis": seasonal_data}

    def get_popular_room_types(self) -> Dict[str, Any]:
        if 'reserved_room_type' not in self.booking_data.columns:
            return {"found": False, "message": "Room type data not available."}
        room_counts = self.booking_data['reserved_room_type'].value_counts()
        total_bookings = len(self.booking_data)
        room_percentages = (room_counts / total_bookings * 100).round(2).to_dict()
        return {
            "found": True,
            "popular_room_types": room_percentages,
            "most_popular": room_counts.idxmax(),
            "total_bookings": total_bookings
        }

    def get_room_type_revenue(self) -> Dict[str, Any]:
        if 'reserved_room_type' not in self.booking_data.columns or 'total_price' not in self.booking_data.columns:
            return {"found": False, "message": "Room type or revenue data not available."}
        room_revenue = self.booking_data.groupby('reserved_room_type')['total_price'].sum().round(2).to_dict()
        return {
            "found": True,
            "revenue_by_room_type": room_revenue,
            "currency": "USD"
        }

    def get_room_type_stats(self) -> Dict[str, Any]:
        if 'reserved_room_type' not in self.booking_data.columns:
            return {"found": False, "message": "Room type data not available."}
        room_counts = self.booking_data['reserved_room_type'].value_counts().to_dict()
        return {
            "found": True,
            "room_type_distribution": room_counts
        }

    def get_length_of_stay_stats(self, segment: Optional[str] = None) -> Dict[str, Any]:
        """
        Get length of stay statistics, optionally filtered by market segment.

        Args:
            segment (str, optional): Market segment to filter by (e.g., 'Corporate', 'Online TA'). Defaults to None (all data).

        Returns:
            Dict[str, Any]: Length of stay statistics.
        """
        print(f"Calling get_length_of_stay_stats with segment: {segment}")
        if segment:
            data = self.booking_data[self.booking_data['market_segment'] == segment]
            if data.empty:
                return {"found": False, "message": f"No data available for segment '{segment}'."}
        else:
            data = self.booking_data

        if 'total_nights' not in data.columns:
            total_nights = data['stays_in_weekend_nights'] + data['stays_in_week_nights']
        else:
            total_nights = data['total_nights']

        stats = {
            "found": True,
            "segment": segment if segment else "All",
            "length_of_stay_stats": {
                "average_length_of_stay": round(total_nights.mean(), 2),
                "median_length_of_stay": round(total_nights.median(), 2),
                "min_length_of_stay": int(total_nights.min()),
                "max_length_of_stay": int(total_nights.max())
            }
        }
        return stats

    def get_booking_channel_stats(self) -> Dict[str, Any]:
        if 'market_segment' not in self.booking_data.columns:
            return {"found": False, "message": "Booking channel data not available."}
        channel_counts = self.booking_data['market_segment'].value_counts()
        total_bookings = len(self.booking_data)
        channel_percentages = (channel_counts / total_bookings * 100).round(2).to_dict()
        return {
            "found": True,
            "booking_channel_distribution": channel_percentages,
            "total_bookings": total_bookings
        }

    def get_general_analytics(self) -> Dict[str, Any]:
        total_bookings = len(self.booking_data)
        total_revenue = (self.booking_data['adr'] * self.booking_data['stays_in_weekend_nights'] +
                         self.booking_data['adr'] * self.booking_data['stays_in_week_nights']).sum()
        cancellation_rate = (self.booking_data['is_canceled'].sum() / total_bookings * 100).round(2)
        return {
            "found": True,
            "total_bookings": total_bookings,
            "total_revenue": round(total_revenue, 2),
            "cancellation_rate": cancellation_rate,
            "average_daily_rate": round(self.booking_data['adr'].mean(), 2),
            "currency": "USD"
        }

    def get_cancellation_patterns(self) -> Dict[str, Any]:
        cancellation_rate_by_month = self.booking_data.groupby('arrival_date_month')['is_canceled'].mean() * 100
        cancellation_rate_by_segment = self.booking_data.groupby('market_segment')['is_canceled'].mean() * 100
        return {
            "found": True,
            "cancellation_rate_by_month": cancellation_rate_by_month.round(2).to_dict(),
            "cancellation_rate_by_segment": cancellation_rate_by_segment.round(2).to_dict(),
            "total_cancellation_rate": round(self.booking_data['is_canceled'].mean() * 100, 2)
        }

    def get_monthly_revenue_by_year(self, month: str, year: int) -> Dict[str, Any]:
        """Get revenue for a specific month and year."""
        month_data = self.booking_data[
            (self.booking_data['arrival_date_month'] == month.title()) &
            (self.booking_data['arrival_date_year'] == year)
            ]
        if month_data.empty:
            return {"found": False, "message": f"No booking data available for {month.title()} {year}."}
        revenue = (month_data['adr'] * month_data['stays_in_weekend_nights'] +
                   month_data['adr'] * month_data['stays_in_week_nights']).sum()
        return {
            "found": True,
            "month": month.title(),
            "year": year,
            "revenue": round(revenue, 2),
            "booking_count": len(month_data),
            "average_daily_rate": round(month_data['adr'].mean(), 2),
            "currency": "USD"
        }

    # def get_highest_cancellation_locations(self) -> Dict[str, Any]:
    #     """Get locations (countries) with the highest cancellation rates."""
    #     if 'country' not in self.booking_data.columns or 'is_canceled' not in self.booking_data.columns:
    #         return {"found": False, "message": "Country or cancellation data not available."}
    #     cancellation_by_country = self.booking_data.groupby('country')['is_canceled'].mean() * 100
    #     total_bookings_by_country = self.booking_data['country'].value_counts()
    #     # Filter for countries with at least 10 bookings to ensure statistical relevance
    #     valid_countries = total_bookings_by_country[total_bookings_by_country >= 10].index
    #     cancellation_by_country = cancellation_by_country[valid_countries].round(2)
    #     top_cancellations = cancellation_by_country.sort_values(ascending=False).head(5).to_dict()
    #     return {
    #         "found": True,
    #         "top_cancellation_locations": top_cancellations,
    #         "total_cancellation_rate": round(self.booking_data['is_canceled'].mean() * 100, 2)
    #     }

    # def get_highest_cancellation_locations(self) -> Dict[str, Any]:
    #     if 'country' not in self.booking_data.columns or 'is_canceled' not in self.booking_data.columns:
    #         return {"found": False, "message": "Country or cancellation data not available."}
    #     cancellation_by_country = self.booking_data.groupby('country')['is_canceled'].mean() * 100
    #     total_bookings_by_country = self.booking_data['country'].value_counts()
    #     # Filter for countries with at least 10 bookings
    #     valid_countries = total_bookings_by_country[total_bookings_by_country >= 10].index
    #     cancellation_by_country = cancellation_by_country[valid_countries].round(2)
    #     top_cancellations = cancellation_by_country.sort_values(ascending=False).head(5).to_dict()
    #     return {
    #         "found": True,
    #         "top_cancellation_locations": top_cancellations,
    #         "total_cancellation_rate": round(self.booking_data['is_canceled'].mean() * 100, 2),
    #         "total_bookings": len(self.booking_data)
    #     }

    def get_highest_cancellation_locations(self) -> Dict[str, Any]:
        if 'country' not in self.booking_data.columns or 'is_canceled' not in self.booking_data.columns:
            return {"found": False, "message": "Country or cancellation data not available."}
        cancellation_by_country = self.booking_data.groupby('country')['is_canceled'].mean() * 100
        total_bookings_by_country = self.booking_data['country'].value_counts()
        valid_countries = total_bookings_by_country[total_bookings_by_country >= 10].index
        cancellation_by_country = cancellation_by_country[valid_countries].round(2)
        top_cancellations = cancellation_by_country.sort_values(ascending=False).head(5).to_dict()
        result = {
            "found": True,
            "top_cancellation_locations": top_cancellations,
            "total_cancellation_rate": round(self.booking_data['is_canceled'].mean() * 100, 2),
            "total_bookings": len(self.booking_data)
        }
        print(f"Debug: {result}")  # Add for debugging
        return result

    # def get_average_booking_price(self) -> Dict[str, Any]:
    #     """Calculate the average price of a hotel booking."""
    #     if 'total_price' not in self.booking_data.columns:
    #         return {"found": False, "message": "Total price data not available."}
    #     avg_price = self.booking_data['total_price'].mean()
    #     return {
    #         "found": True,
    #         "average_booking_price": round(avg_price, 2),
    #         "total_bookings": len(self.booking_data),
    #         "currency": "USD"
    #     }

    def get_average_booking_price(self) -> Dict[str, Any]:
        if 'total_price' not in self.booking_data.columns:
            return {"found": False, "message": "Total price data not available."}
        avg_price = self.booking_data['total_price'].mean()
        return {
            "found": True,
            "average_booking_price": round(avg_price, 2),
            "total_bookings": len(self.booking_data),
            "currency": "USD"
        }

    def query_with_rag(self, query: str) -> Dict[str, Any]:
        context = "\n\n".join([doc['text'] for doc in self.retrieve_relevant_context(query)])
        prompt = f"Question: {query}\nContext: {context}\nAnswer:"
        response = self.llm(prompt, max_length=200, num_return_sequences=1)[0]['generated_text']
        return {
            "found": True,
            "response": response,
            "sources_used": [{"type": "booking_data", "date_range": "all", "relevance_score": 0.9}]
        }

# Updated HotelAnalyticsSystem class
class HotelAnalyticsSystem:
    def init(self, booking_data_path: str, total_rooms: int = 120):
        """
        Initialize the Hotel Analytics System.

        Args:
            booking_data_path: Path to the CSV file containing booking data
            total_rooms: Total number of rooms in the hotel
        """
        # Initialize DataProcessor
        self.data_processor = DataProcessor(booking_data_path)
        self.data_processor.preprocess_data()
        self.data_processor.initialize_database()
        self.data_processor.store_data_in_db()

        # Load booking data
        self.booking_data = self.data_processor.processed_data
        print(self.booking_data['arrival_date_month'].unique())
        self.total_rooms = total_rooms

        # Initialize RAG engine with total_rooms
        self.rag_engine = RAGEngine(self.data_processor)
        self.rag_engine.total_rooms = total_rooms  # Pass total_rooms to RAGEngine
        self.rag_engine.initialize_embedding_model()
        self.rag_engine.prepare_text_data()
        self.rag_engine.build_faiss_index()

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a natural language query about hotel bookings.
        Delegates to RAGEngine's handle_analytics_query.
        """
        return self.rag_engine.handle_analytics_query(query)

    def generate_dashboard(self) -> Dict[str, Any]:
        """Generate a comprehensive dashboard with key metrics."""
        if self.booking_data is None:
            return {"error": "No data available for dashboard generation."}

        current_year = max(self.booking_data['arrival_date_year'])
        total_bookings = len(self.booking_data)
        average_daily_rate = self.booking_data['adr'].mean()
        average_length_of_stay = self.booking_data['total_nights'].mean()
        monthly_bookings = self.booking_data.groupby('arrival_date_month').size()
        month_order = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
            'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        monthly_bookings = monthly_bookings.reindex(
            sorted(monthly_bookings.index, key=lambda x: month_order.get(x, 13)))
        room_type_distribution = (self.booking_data['reserved_room_type'].value_counts() / total_bookings * 100).round(
            2)
        market_segment_distribution = (self.booking_data['market_segment'].value_counts() / total_bookings * 100).round(
            2)
        cancellation_rate = (self.booking_data['is_canceled'].sum() / total_bookings * 100).round(2)

        return {
            "total_bookings": int(total_bookings),
            "average_daily_rate": round(average_daily_rate, 2),
            "average_length_of_stay": round(average_length_of_stay, 2),
            "cancellation_rate": round(cancellation_rate, 2),
            "monthly_bookings": monthly_bookings.to_dict(),
            "room_type_distribution": room_type_distribution.to_dict(),
            "market_segment_distribution": market_segment_distribution.to_dict(),
            "currency": "USD"
        }

    # def evaluate_performance(self) -> Dict[str, Any]:
    #     """Evaluate Q&A accuracy and API response time."""
    #     test_queries = {
    #         "What was our revenue in July?": {"revenue": 53186.87},  # Adjust based on your data
    #         "What's the average price of a hotel booking?": None  # Dynamic, just check if found
    #     }
    #     accuracy_results = []
    #     response_times = []
    #
    #     for query, expected in test_queries.items():
    #         start_time = time.time()
    #         result = self.process_query(query)
    #         response_time = time.time() - start_time
    #         response_times.append(response_time)
    #
    #         if expected:
    #             is_correct = abs(result.get("revenue", 0) - expected["revenue"]) < 0.01
    #             accuracy_results.append(is_correct)
    #         else:
    #             accuracy_results.append(result.get("found", False))
    #
    #     accuracy = sum(accuracy_results) / len(accuracy_results) * 100
    #     avg_response_time = sum(response_times) / len(response_times)
    #
    #     return {
    #         "qna_accuracy": round(accuracy, 2),
    #         "average_response_time_seconds": round(avg_response_time, 2),
    #         "test_queries_evaluated": len(test_queries)
    #     }
    def evaluate_performance(self) -> Dict[str, Any]:
        import time  # Ensure this is included
        test_queries = {
            "What was our revenue in July?": {"revenue": 53186.87},  # Adjust based on your data
            "What's the average price of a hotel booking?": None  # Dynamic, just check if found
        }
        accuracy_results = []
        response_times = []

        for query, expected in test_queries.items():
            start_time = time.time()
            result = self.process_query(query)
            response_time = time.time() - start_time
            response_times.append(response_time)

            if expected:
                # Check if the result contains the expected key and matches the value within a tolerance
                revenue = result.get("revenue", 0) if isinstance(result, dict) else 0
                is_correct = abs(revenue - expected["revenue"]) < 0.01  # Small tolerance for float comparison
                accuracy_results.append(is_correct)
            else:
                # For queries without specific expected values, check if a result was found
                is_found = result.get("found", False) if isinstance(result, dict) else False
                accuracy_results.append(is_found)

        accuracy = sum(accuracy_results) / len(accuracy_results) * 100
        avg_response_time = sum(response_times) / len(response_times)

        return {
            "qna_accuracy": round(accuracy, 2),
            "average_response_time_seconds": round(avg_response_time, 2),
            "test_queries_evaluated": len(test_queries)
        }

