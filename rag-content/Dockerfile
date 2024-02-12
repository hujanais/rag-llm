FROM python:3.11-slim

WORKDIR /usr/src/app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 5000

CMD [ "python3", "-m" , "flask", "--app", "./server/app.py", "run", "--host=0.0.0.0"]