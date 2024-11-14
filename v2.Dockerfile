FROM mgrcar/deepface:0.1

RUN pip install fastapi
RUN pip install uvicorn
RUN pip install python-multipart

ENTRYPOINT ["python3"]
CMD ["/files/server.py"]