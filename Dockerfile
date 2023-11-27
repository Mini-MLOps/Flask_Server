FROM python:3.9-slim-buster
COPY . /app
WORKDIR /app
RUN python3 -m venv venv && . venv/bin/activate 
RUN pip3 install -r packagelist.txt
ENTRYPOINT ["python3", "-m", "flask", "run"]
CMD ["--host=0.0.0.0"]