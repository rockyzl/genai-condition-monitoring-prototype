# Hugging Face Spaces (docker SDK) container for the public demo.
# HF no longer offers a native streamlit SDK for new Spaces, so we run
# Streamlit inside the standard HF docker pattern (non-root user 1000,
# app served on the port declared as app_port in the Space README).
FROM python:3.12-slim

WORKDIR /app

RUN useradd -m -u 1000 user && chown user:user /app
COPY --chown=user:user . /app
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    MPLCONFIGDIR=/tmp/mpl

RUN pip install --no-cache-dir --user -r requirements.txt

EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
