FROM python:3.11-slim
WORKDIR /app
COPY requirements/base.txt ./requirements/base.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements/base.txt
COPY . .
RUN mkdir -p /app/data/production && chmod -R 755 /app && chmod 775 /app/data/production
EXPOSE 7860
ENV PORT=7860
CMD ["python", "run_nourai.py", "--no-browser", "--host", "0.0.0.0"]
