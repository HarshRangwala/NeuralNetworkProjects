## Genrate Music using Multi-Layer LSTM

### Requirements
> tensorflow <br>
> keras <br>
> numpy <br>
> matplotlib <br>
> pickle <br>
> music21 <br>

### Execution
Make sure all the files are set with right directories. <br>
Run <b>training.py</b> to train the model.<br> <br>
```python3 training.py``` <br> <br>
After training phase is complete, load the learned model weight in <b>predict.py</b> and then run using following command. <br> <br>
```python3 predict.py``` <br> <br>
You will find the generated output in ```generated_music``` folder. <br> <br>
Thats it!! <br> <br>

### Reference
[next.tech for an awesome tutorial](http://next.tech/) <br>
[Understanding LSTM networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
