## Genrate Music using Multi-Layer LSTM

### Requirements
> tensorflow
> keras
> numpy
> matplotlib
> pickle
> music21

### Execution
Make sure all the files are set with right directories. <br>
Run <b>training.py</b> to train the model.<br>
```python3 training.py``` <br>
After training phase is complete, load the learned model weight in <b>predict.py</b> and then run using following command. <br>
```python3 predict.py``` <br>
You will find the generated output in ```generated_music``` folder. <br>
Thats it!! <br>

### Credits
next.tech for an awesome tutorial
[Understanding LSTM networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
