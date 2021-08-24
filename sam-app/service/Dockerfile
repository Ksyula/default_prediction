# multi-stage build

FROM public.ecr.aws/lambda/python:3.9 as base

FROM base AS train
COPY requirements.txt .
RUN pip install -r requirements.txt
ENV MODEL_LOCAL_PATH=pickled_model.pickle
COPY pickled_model.pickle .

FROM base
RUN pip install scikit-learn==0.22.1
ENV MODEL_LOCAL_PATH=pickled_model.pickle
COPY --from=train ./pickled_model.pickle pickled_model.pkl
COPY app.py ./


CMD ["app.lambda_handler"]