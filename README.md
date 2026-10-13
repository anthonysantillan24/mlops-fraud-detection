# Real-Time Fraud Detection MLOps Pipeline

Production-ready end-to-end MLOps pipeline designed for real-time transaction fraud detection. The project integrates machine learning model inference with a high-performance FastAPI web service, fully containerized using Docker and Docker Compose to ensure reproducibility and seamless deployment across environments.

## Technical Architecture & Tools

* **Programming Language:** Python
* **Machine Learning:** Scikit-Learn
* **API Framework:** FastAPI & Pydantic (data validation and serialization)
* **Containerization:** Docker & Docker Compose
* **Testing:** Pytest
* **Version Control:** Git & GitHub

## Project Structure

```text
mlops-fraud-detection/
│
├── src/
│   ├── __init__.py
│   ├── train.py        # Model training pipeline and artifact generation
│   ├── predict.py      # Core inference logic
│   └── api.py          # RESTful API endpoints and application routing
│
├── tests/
│   └── __init__.py     # Automated unit and integration test suite
│
├── .gitignore          # Version control exclusion rules
├── Dockerfile          # Container build specifications
├── docker-compose.yml  # Service orchestration configuration
└── requirements.txt    # Dependency specifications

Setup and Execution
Prerequisites
Docker Engine

Docker Compose

Git

Local Deployment via Docker
Clone the repository:

Bash
git clone [https://github.com/anthonysantillan24/mlops-fraud-detection.git](https://github.com/anthonysantillan24/mlops-fraud-detection.git)
cd mlops-fraud-detection
Build and launch the containerized service:

Bash
docker compose up --build
Interactive API Documentation:
Once the service is active, access the Swagger UI documentation at:

Plaintext
http://localhost:8000/docs
