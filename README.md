# Hotel Analytics System

A Python-based hotel analytics system integrating data processing, retrieval-augmented generation (RAG) with a language model, and analytics capabilities to answer queries about hotel booking data.

---

## Overview

This project provides a comprehensive solution for analyzing hotel booking data. It includes:
- **Data Processing**: Loads, preprocesses, and stores booking data in SQLite.
- **Analytics Engine**: Computes key metrics like revenue trends and lead time distributions.
- **RAG Engine**: Uses FAISS for vector search and an LLM (GPT-Neo) for natural language query responses.
- **API**: (Optional) FastAPI integration for future web-based access.
- **Evaluation**: Performance testing for accuracy and response time.

---

## Setup Instructions

### Prerequisites
- **Python**: Version 3.7 or higher (tested with 3.13).
- **Git**: To clone the repository.
- **VS Code**: Recommended IDE (optional).

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/hotel-analytics-system.git
   cd hotel-analytics-system

2. **Create a Virtual Environment**
   - **On Windows:**
       ```bash
       python -m venv venv
       .\venv\Scripts\activate
   
   - **On macOS/Linux:**
   
       ```bash
       python3 -m venv venv
       source venv/bin/activate

3. **Install Dependencies**
   - Update pip to the latest version:
       ```bash
       python -m pip install --upgrade pip

   - Install required packages:
       ```bash
       pip install -r requirements.txt

   - If you encounter timeouts, increase the timeout:
       ```bash
       pip install -r requirements.txt --timeout 100

4. **Download Sample Data**
   - Place a hotel_bookings.csv file in the data/ directory (e.g., from Kaggle Hotel Booking Demand).
   - Alternatively, use the provided sample data/sample_hotel_bookings.csv.

5. **Run the Application**
   - Execute the main script:
       ```bash
       python hotel_analytics.py

   - Test queries interactively or use the provided test suite (see below).

6. **(Optional) Run the API**
   - Start the FastAPI server:
       ```bash
       uvicorn hotel_analytics_api:app --reload
   - Access at http://127.0.0.1:8000.

## GitHub Repository Structure
    
    hotel-analytics-system/
    ├── data/
    │   ├── sample_hotel_bookings.csv   # Sample dataset
    │   └── hotel_bookings.db           # SQLite database (generated)
    ├── models/                         # Directory for pre-trained models (if any)
    ├── tests/
    │   ├── test_queries.json           # Sample test queries and expected answers
    │   └── test_hotel_analytics.py     # Test script
    ├── hotel_analytics.py              # Main codebase (LLM, analytics, API)
    ├── hotel_analytics_api.py          # FastAPI implementation
    ├── requirements.txt                # Dependencies
    └── README.md                       # This file 

### Files Description
- hotel_analytics.py: Core logic including DataProcessor, AnalyticsEngine, RAGEngine, and HotelAnalyticsSystem.
- hotel_analytics_api.py: FastAPI wrapper (currently unused but included for expansion).
- requirements.txt: List of Python dependencies.
- tests/test_queries.json: JSON file with sample queries and expected outputs.
- data/sample_hotel_bookings.csv: Subset of booking data for testing.

## Sample Test Queries & Expected Answers
Stored in tests/test_queries.json:
```json
    {
  "queries": [
    {
      "query": "What was our revenue in July?",
      "expected": {
        "found": true,
        "revenue": 53186.87,
        "currency": "USD"
      }
    },
    {
      "query": "What's the average price of a hotel booking?",
      "expected": {
        "found": true,
        "average_booking_price": 325.50,
        "total_bookings": 1000,
        "currency": "USD"
      }
    },
    {
      "query": "What’s the lead time distribution?",
      "expected": {
        "found": true,
        "lead_time_stats": {
          "mean": 104.01,
          "median": 69.0,
          "min": 0,
          "max": 737
        }
      }
    }
  ]
}

```

- Notes:
    - Expected values (e.g., revenue, averages) depend on your     dataset. Adjust these based on data/sample_hotel_bookings.csv.
    - Run tests/test_hotel_analytics.py to evaluate:
    ```bash
    python tests/test_hotel_analytics.py

## Implementation Choices & Challenges
### Implementation Choices
1. Modular Design:
   - Separated into DataProcessor, AnalyticsEngine, and RAGEngine for maintainability and scalability.
   - Allows independent testing and future enhancements (e.g., adding new analytics).
2. RAG with LLM:
   - Used FAISS for efficient vector search of booking data context.
   - Integrated GPT-Neo-125M (via transformers) for natural language responses due to its lightweight nature and open-source availability.
3. SQLite Database:
   - Chosen for lightweight storage and querying of preprocessed data, avoiding repeated CSV parsing.
4. FastAPI:
   - Included as an optional API layer for potential web deployment, though not fully utilized in this version.
5. Dependencies:
   - Selected pandas for data manipulation, sentence-transformers for embeddings, and faiss-cpu for vector indexing due to their robustness and community support.
### Challenges
1. LLM Integration:
   - Challenge: GPT-Neo-125M generates verbose or off-topic responses without fine-tuning.
   - Solution: Combined rule-based analytics with LLM for structured outputs, using RAG to provide relevant context.
2. Dependency Installation:
   - Challenge: Building pandas from source caused timeouts or compilation errors on some systems.
   - Solution: Recommended virtual environments and precompiled wheels; added timeout options in setup instructions.
3. Data Preprocessing:
   - Challenge: Handling missing or inconsistent date formats in booking data.
   - Solution: Robust datetime parsing with errors='coerce' and median/mode imputation for missing values.
4. Performance:
   - Challenge: Embedding large datasets with FAISS slowed initialization.
   - Solution: Sampled data (1000 rows) for testing; full dataset support requires optimization (e.g., batch processing).
5. Scalability:
   - Challenge: Current design processes queries synchronously.
   - Future Work: Add asynchronous query handling with FastAPI or multiprocessing for larger datasets.
## Contributing
- Fork the repository, make changes, and submit a pull request.
- Report issues or suggest features via GitHub Issues.
## License
MIT License - feel free to use and modify this code.
