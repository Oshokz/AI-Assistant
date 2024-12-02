FROM python:3.13.0
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD streamlit run app.py