# AI Outfit Recommender

AI-based fashion outfit recommendation system developed using Python and Streamlit.

## Requirements

Before running the project, make sure the following are installed:

- Python 3.10 or later
- Git
- Internet connection

## Installation

### 1. Clone the Repository

Open Command Prompt or Terminal and run:

bash
git clone https://github.com/cu00cu/HnM-Outfit-Recommender.git
cd HnM-Outfit-Recommender


### 2. Create a Virtual Environment

bash
python -m venv venv


### 3. Activate the Virtual Environment

For Windows:

bash
venv\Scripts\activate


For macOS/Linux:

bash
source venv/bin/activate


### 4. Install Required Python Libraries

Install all required libraries using the provided requirements.txt file:

bash
pip install -r requirements.txt


### 5. Run the Application

Run the Streamlit application:

bash
streamlit run app.py


The application will normally be available at:

text
http://localhost:8501


Open the URL in a web browser to use the application.

## Project Structure

text
HnM-Outfit-Recommender/
├── sample_data/
├── app.py
├── requirements.txt
└── README.md


## Dataset

The application uses the H&M Personalized Fashion Recommendations dataset.

The required sample data is provided in the sample_data folder.

For the complete H&M dataset, visit:

https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations

## Important Notes

- Make sure the virtual environment is activated before running the application.
- Make sure all required libraries are installed using requirements.txt.
- An internet connection may be required when downloading AI models for the first time.
- The first run may take longer because the required AI models may need to be downloaded.
- Do not rename or move the files in sample_data unless the corresponding paths in app.py are also updated.
