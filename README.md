# CLAF: Chord Learning and Adversarial Framework for Symbolic Polyphonic Music Generation

> you can find samples of generated music (selected or random)at <https://ryf1123.github.io/>

#### dependencies

```
python3
PyTorch

```

#### Train the model

##### Element-wise Reserved model

```shell
python3 main.py --model "ew"
```

##### Discriminator Dominant  model

```shell
python3 main.py --model "dd"
```

##### Hybrid model

```shell
python3 main.py --model "hybrid"
```

Then, trained model will be saved at `./saved_model`

#### Generate from pretrained model

##### Element-wise reserved model

```shell
python3 main.py --model "ew" --pretrained 
```

Then, the model will generate from the pretrained model, which you can get from this [link]().

##### Discriminator Dominant model

```shell
python3 main.py --model "dd" --pretrained 
```

##### Hybrid model

```shell
python3 main.py --model "hybrid" --pretrained 
```

#### File tree













