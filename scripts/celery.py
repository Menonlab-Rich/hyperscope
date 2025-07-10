from celery import Celery

app = Celery('hello', broker='amqp://user@localhost//')

filenames = ['a', 'b', 'c', 'd']

@app.task
def hello():
    return 'hello world'

@app.task(bind=True)
def upload_files(self, filenames):
    for i, file in enumerate(filenames):
        if not self.request.called_directly:
            self.update_state(state='PROGRESS',
                meta={'current': i, 'total': len(filenames)})

