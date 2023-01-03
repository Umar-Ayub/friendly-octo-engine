FROM python:3.9.15

ARG APPDIR=/app 
WORKDIR ${APPDIR}

COPY requirements.txt ${APPDIR}

RUN python3 -m pip install -r ${APPDIR}/requirements.txt

ARG LOCATION
ENV MODEL_LOCATION=${APPDIR}/model.pkl

COPY serve.py ${APPDIR}
COPY ${LOCATION} ${APPDIR}

CMD [ "python", "serve.py" ]